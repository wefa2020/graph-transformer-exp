"""
data/dataset.py - Dataset with S3 H5 cache support for Causal Graph Transformer with Time2Vec
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Any, Optional, Callable, List
import pickle
import multiprocessing as mp
import h5py
import os
import tempfile


# =============================================================================
# TENSOR CONVERSION
# =============================================================================

def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert numpy array to tensor with specified dtype."""
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


# =============================================================================
# WORKER FUNCTIONS FOR PARALLEL PROCESSING
# =============================================================================

_worker_preprocessor = None


def _init_worker(preprocessor_bytes: bytes):
    """Initialize worker with preprocessor."""
    global _worker_preprocessor
    _worker_preprocessor = pickle.loads(preprocessor_bytes)


def _process_item(args):
    """Process a single item in worker."""
    global _worker_preprocessor
    idx, row_dict = args
    try:
        features = _worker_preprocessor.process_lifecycle(row_dict, return_labels=True)
        return (idx, features, None)
    except Exception as e:
        return (idx, None, str(e))


# =============================================================================
# S3 UTILITIES
# =============================================================================

def is_s3_path(path: str) -> bool:
    """Check if path is an S3 path."""
    return path is not None and str(path).startswith('s3://')


def parse_s3_path(s3_path: str) -> tuple:
    """Parse S3 path into bucket and key."""
    path = s3_path[5:] if s3_path.startswith('s3://') else s3_path
    parts = path.split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def s3_exists(s3_path: str) -> bool:
    """Check if S3 object exists."""
    try:
        import boto3
        bucket, key = parse_s3_path(s3_path)
        boto3.client('s3').head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def s3_download(s3_path: str, local_path: str, log_fn: Callable = None):
    """Download file from S3."""
    import boto3
    from boto3.s3.transfer import TransferConfig
    
    bucket, key = parse_s3_path(s3_path)
    if log_fn:
        log_fn(f"  Downloading s3://{bucket}/{key}")
    
    os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
    
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,
    )
    
    boto3.client('s3').download_file(bucket, key, local_path, Config=config)
    
    if log_fn:
        log_fn(f"  ✓ Downloaded {os.path.getsize(local_path) / 1e6:.1f} MB")


def s3_upload(local_path: str, s3_path: str, log_fn: Callable = None):
    """Upload file to S3."""
    import boto3
    from boto3.s3.transfer import TransferConfig
    
    bucket, key = parse_s3_path(s3_path)
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    
    if log_fn:
        log_fn(f"  Uploading to s3://{bucket}/{key} ({file_size_mb:.1f} MB)")
    
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,
    )
    
    boto3.client('s3').upload_file(local_path, bucket, key, Config=config)
    
    if log_fn:
        log_fn(f"  ✓ Uploaded to s3://{bucket}/{key}")


# =============================================================================
# DATASET
# =============================================================================

class PackageLifecycleDataset(Dataset):
    """
    Dataset for Causal Graph Transformer with Time2Vec and S3 H5 cache support.
    
    Key features:
    - Separates time features (for Time2Vec) from other features
    - Separates OBSERVABLE and REALIZED node features
    - Computes causal masks on-the-fly (deterministic)
    - Efficient H5 storage with S3 support
    - Parallel preprocessing with multiprocessing
    
    Feature Structure:
    - node_observable_time: [N, 6] - hour, dow, dom, month, elapsed, time_until_plan
    - node_observable_other: [N, 3] - is_delivery, position, has_plan
    - node_realized_time: [N, 6] - hour, dow, dom, month, elapsed, time_vs_plan
    - node_realized_other: [N, 5+num_problems] - time_since_prev, dwelling, etc.
    - edge_features: [E, 8] - distance, flags, hour, dow, dom, month
    """
    
    # Node categorical fields (7 features)
    _NODE_CAT_FIELDS = [
        'event_type', 'location', 'postal', 'region',
        'carrier', 'leg_type', 'ship_method'
    ]
    
    def __init__(
        self,
        df=None,
        preprocessor=None,
        h5_cache_path: str = None,
        load_from_cache: bool = False,
        save_to_cache: bool = True,
        num_workers: Optional[int] = None,
        log_fn: Callable = None
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with 'events' column
            preprocessor: Fitted PackageLifecyclePreprocessor
            h5_cache_path: S3 or local path for H5 cache
            load_from_cache: Whether to load from existing cache
            save_to_cache: Whether to save processed data to cache
            num_workers: Number of parallel workers
            log_fn: Logging function
        """
        self._log = log_fn or (lambda x: None)
        self._num_workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
        
        self._s3_path: str = None
        self._local_path: str = None
        self._node_offsets: np.ndarray = None
        self._edge_offsets: np.ndarray = None
        self._num_samples: int = 0
        self._has_labels: bool = False
        self._collator: Optional['CausalH5BatchCollator'] = None
        self._temp_file: bool = False
        
        # Feature dimensions (set after loading) - Updated for Time2Vec
        self._observable_time_dim: int = 6
        self._observable_other_dim: int = 3
        self._realized_time_dim: int = 6
        self._realized_other_dim: int = 0
        self._edge_dim: int = 8
        self._package_dim: int = 4
        
        self._log(f"PackageLifecycleDataset: {h5_cache_path}")
        
        # Determine if S3 path
        is_s3 = is_s3_path(h5_cache_path) if h5_cache_path else False
        
        # Try to load from cache
        if load_from_cache and h5_cache_path:
            if is_s3:
                if s3_exists(h5_cache_path):
                    self._init_from_s3(h5_cache_path)
                    return
                else:
                    self._log(f"  S3 cache not found: {h5_cache_path}")
            else:
                if os.path.exists(h5_cache_path):
                    self._init_from_local(h5_cache_path)
                    return
                else:
                    self._log(f"  Local cache not found: {h5_cache_path}")
            
            self._log(f"  Cache not found, will create new")
        
        # Process DataFrame
        if df is not None and preprocessor is not None:
            if not preprocessor.fitted:
                raise ValueError("Preprocessor must be fitted")
            
            cache = self._process_dataframe(df, preprocessor)
            
            if save_to_cache and h5_cache_path:
                if is_s3:
                    self._save_and_upload_to_s3(cache, h5_cache_path)
                else:
                    os.makedirs(os.path.dirname(h5_cache_path) or '.', exist_ok=True)
                    self._write_h5(cache, h5_cache_path)
                    self._init_from_local(h5_cache_path)
            else:
                self._init_from_cache(cache)
        elif not load_from_cache:
            raise ValueError("Provide (df, preprocessor) or set load_from_cache=True")
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> int:
        return idx
    
    def __del__(self):
        self.close()
        if self._temp_file and self._local_path and os.path.exists(self._local_path):
            try:
                os.unlink(self._local_path)
            except:
                pass
    
    def close(self):
        """Close resources."""
        if self._collator:
            self._collator.close()
            self._collator = None
    
    def get_collate_fn(self) -> 'CausalH5BatchCollator':
        """Get collate function for DataLoader."""
        if self._collator:
            self._collator.close()
        self._collator = CausalH5BatchCollator(
            h5_path=self._local_path,
            node_offsets=self._node_offsets,
            edge_offsets=self._edge_offsets,
            has_labels=self._has_labels,
            node_cat_fields=self._NODE_CAT_FIELDS,
        )
        return self._collator
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for model initialization."""
        return {
            'observable_time_dim': self._observable_time_dim,
            'observable_other_dim': self._observable_other_dim,
            'realized_time_dim': self._realized_time_dim,
            'realized_other_dim': self._realized_other_dim,
            'edge_dim': self._edge_dim,
            'package_dim': self._package_dim,
        }
    
    def _init_from_s3(self, s3_path: str):
        """Download from S3 and initialize."""
        self._s3_path = s3_path
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        s3_download(s3_path, self._local_path, self._log)
        self._read_h5_metadata()
    
    def _init_from_local(self, local_path: str):
        """Initialize from local H5 file."""
        self._local_path = local_path
        self._temp_file = False
        self._read_h5_metadata()
    
    def _init_from_cache(self, cache: List[Dict]):
        """Initialize from in-memory cache (for small datasets)."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        self._write_h5(cache, self._local_path)
        self._read_h5_metadata()
    
    def _save_and_upload_to_s3(self, cache: List[Dict], s3_path: str):
        """Save H5 locally and upload to S3."""
        self._s3_path = s3_path
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        self._write_h5(cache, self._local_path)
        s3_upload(self._local_path, s3_path, self._log)
        self._read_h5_metadata()
    
    def _read_h5_metadata(self):
        """Read metadata from H5 file."""
        with h5py.File(self._local_path, 'r') as f:
            self._num_samples = int(f.attrs['num_samples'])
            self._has_labels = bool(f.attrs.get('has_labels', False))
            
            # Time2Vec feature dimensions
            self._observable_time_dim = int(f.attrs.get('observable_time_dim', 6))
            self._observable_other_dim = int(f.attrs.get('observable_other_dim', 3))
            self._realized_time_dim = int(f.attrs.get('realized_time_dim', 6))
            self._realized_other_dim = int(f.attrs.get('realized_other_dim', 20))
            self._edge_dim = int(f.attrs.get('edge_dim', 8))
            self._package_dim = int(f.attrs.get('package_dim', 4))
            
            self._node_offsets = f['node_offsets'][:]
            self._edge_offsets = f['edge_offsets'][:]
        
        self._log(f"  Loaded {self._num_samples} samples")
        self._log(f"  Time2Vec dims: obs_time={self._observable_time_dim}, obs_other={self._observable_other_dim}")
        self._log(f"  Time2Vec dims: real_time={self._realized_time_dim}, real_other={self._realized_other_dim}")
        self._log(f"  Edge dim={self._edge_dim}, Package dim={self._package_dim}")
    
    def _process_dataframe(self, df, preprocessor) -> List[Dict]:
        """Process DataFrame in parallel."""
        preprocessor_bytes = pickle.dumps(preprocessor)
        work_items = [(idx, row.to_dict()) for idx, (_, row) in enumerate(df.iterrows())]
        total = len(work_items)
        
        self._log(f"  Processing {total} samples with {self._num_workers} workers...")
        
        results_dict = {}
        errors = []
        
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context('spawn')
        
        with ctx.Pool(self._num_workers, initializer=_init_worker,
                      initargs=(preprocessor_bytes,), maxtasksperchild=1000) as pool:
            for i, (idx, features, error) in enumerate(pool.imap_unordered(_process_item, work_items, chunksize=2000)):
                if features is not None:
                    if features['num_nodes'] >= 2:
                        results_dict[idx] = features
                elif len(errors) < 10:
                    errors.append((idx, error))
                if (i + 1) % max(1, total // 5) == 0:
                    self._log(f"    Processed {i + 1}/{total}")
        
        if errors:
            self._log(f"  Errors ({len(errors)}): {errors[:3]}")
        
        self._log(f"  Done: {len(results_dict)} valid samples")
        return [results_dict[i] for i in sorted(results_dict.keys())]
    
    def _write_h5(self, cache: List[Dict], path: str):
        """Write H5 file with Time2Vec-compatible feature structure."""
        n_samples = len(cache)
        
        if n_samples == 0:
            raise ValueError("No samples to write")
        
        # Calculate offsets
        node_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        edge_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        
        for i, data in enumerate(cache):
            node_offsets[i + 1] = node_offsets[i] + data['num_nodes']
            edge_offsets[i + 1] = edge_offsets[i] + data['edge_index'].shape[1]
        
        total_nodes = int(node_offsets[-1])
        total_edges = int(edge_offsets[-1])
        
        # Get dimensions from first sample
        sample0 = cache[0]
        
        # Time2Vec feature dimensions
        observable_time_dim = sample0['node_observable_time'].shape[1]
        observable_other_dim = sample0['node_observable_other'].shape[1]
        realized_time_dim = sample0['node_realized_time'].shape[1]
        realized_other_dim = sample0['node_realized_other'].shape[1]
        edge_dim = sample0['edge_features'].shape[1]
        package_dim = sample0['package_features'].shape[0]
        has_labels = 'labels' in sample0
        
        self._log(f"  Writing H5: {total_nodes:,} nodes, {total_edges:,} edges")
        self._log(f"  Time2Vec dims: obs_time={observable_time_dim}, obs_other={observable_other_dim}")
        self._log(f"  Time2Vec dims: real_time={realized_time_dim}, real_other={realized_other_dim}")
        
        # Allocate arrays - Time2Vec structure
        node_observable_time = np.zeros((total_nodes, observable_time_dim), dtype=np.float32)
        node_observable_other = np.zeros((total_nodes, observable_other_dim), dtype=np.float32)
        node_realized_time = np.zeros((total_nodes, realized_time_dim), dtype=np.float32)
        node_realized_other = np.zeros((total_nodes, realized_other_dim), dtype=np.float32)
        
        node_cat = {f: np.zeros(total_nodes, dtype=np.int32) for f in self._NODE_CAT_FIELDS}
        
        edge_index = np.zeros((2, total_edges), dtype=np.int64)
        edge_features = np.zeros((total_edges, edge_dim), dtype=np.float32)
        
        package_features = np.zeros((n_samples, package_dim), dtype=np.float32)
        source_postal = np.zeros(n_samples, dtype=np.int32)
        dest_postal = np.zeros(n_samples, dtype=np.int32)
        
        if has_labels:
            edge_labels = np.zeros(total_edges, dtype=np.float32)
            edge_labels_raw = np.zeros(total_edges, dtype=np.float32)
        
        # Fill arrays
        for i, data in enumerate(cache):
            n_s, n_e = int(node_offsets[i]), int(node_offsets[i + 1])
            e_s, e_e = int(edge_offsets[i]), int(edge_offsets[i + 1])
            
            # Node features - Time2Vec structure
            node_observable_time[n_s:n_e] = data['node_observable_time']
            node_observable_other[n_s:n_e] = data['node_observable_other']
            node_realized_time[n_s:n_e] = data['node_realized_time']
            node_realized_other[n_s:n_e] = data['node_realized_other']
            
            # Node categorical
            for f in self._NODE_CAT_FIELDS:
                node_cat[f][n_s:n_e] = data['node_categorical_indices'][f]
            
            # Edge features (adjust indices for global indexing)
            edge_index[:, e_s:e_e] = data['edge_index'] + n_s
            edge_features[e_s:e_e] = data['edge_features']
            
            # Package features
            package_features[i] = data['package_features']
            source_postal[i] = data['package_categorical']['source_postal']
            dest_postal[i] = data['package_categorical']['dest_postal']
            
            # Labels
            if has_labels:
                labels_data = data['labels']
                if labels_data.ndim > 1:
                    labels_data = labels_data.flatten()
                edge_labels[e_s:e_e] = labels_data
                
                labels_raw_data = data.get('labels_raw', data['labels'])
                if labels_raw_data.ndim > 1:
                    labels_raw_data = labels_raw_data.flatten()
                edge_labels_raw[e_s:e_e] = labels_raw_data
        
        # Write to H5
        with h5py.File(path, 'w') as f:
            # Metadata
            f.attrs['num_samples'] = n_samples
            f.attrs['total_nodes'] = total_nodes
            f.attrs['total_edges'] = total_edges
            f.attrs['has_labels'] = has_labels
            
            # Time2Vec dimensions
            f.attrs['observable_time_dim'] = observable_time_dim
            f.attrs['observable_other_dim'] = observable_other_dim
            f.attrs['realized_time_dim'] = realized_time_dim
            f.attrs['realized_other_dim'] = realized_other_dim
            f.attrs['edge_dim'] = edge_dim
            f.attrs['package_dim'] = package_dim
            
            # Offsets
            f.create_dataset('node_offsets', data=node_offsets)
            f.create_dataset('edge_offsets', data=edge_offsets)
            
            # Node features - Time2Vec structure
            f.create_dataset('node_observable_time', data=node_observable_time,
                           chunks=(min(10000, max(1, total_nodes)), observable_time_dim))
            f.create_dataset('node_observable_other', data=node_observable_other,
                           chunks=(min(10000, max(1, total_nodes)), observable_other_dim))
            f.create_dataset('node_realized_time', data=node_realized_time,
                           chunks=(min(10000, max(1, total_nodes)), realized_time_dim))
            f.create_dataset('node_realized_other', data=node_realized_other,
                           chunks=(min(10000, max(1, total_nodes)), realized_other_dim))
            
            # Node categorical
            grp = f.create_group('node_categorical')
            for k, v in node_cat.items():
                grp.create_dataset(k, data=v)
            
            # Edge features
            f.create_dataset('edge_index', data=edge_index)
            f.create_dataset('edge_features', data=edge_features,
                           chunks=(min(10000, max(1, total_edges)), edge_dim))
            
            # Package features
            f.create_dataset('package_features', data=package_features)
            f.create_dataset('source_postal', data=source_postal)
            f.create_dataset('dest_postal', data=dest_postal)
            
            # Labels
            if has_labels:
                f.create_dataset('edge_labels', data=edge_labels)
                f.create_dataset('edge_labels_raw', data=edge_labels_raw)
        
        self._log(f"  ✓ H5 written: {os.path.getsize(path) / 1e6:.1f} MB")


# =============================================================================
# BATCH COLLATOR
# =============================================================================

class CausalH5BatchCollator:
    """
    Collate function that reads batch from H5 file with Time2Vec feature structure.
    
    Produces batches with:
    - node_observable_time: [N, 6] - time features for Time2Vec
    - node_observable_other: [N, 3] - other observable features
    - node_realized_time: [N, 6] - realized time features for Time2Vec
    - node_realized_other: [N, realized_other_dim] - other realized features
    - edge_features: [E, 8] - edge features (includes time for EdgeTime2Vec)
    - causal_masks: List of tensors for causal masking
    """
    
    def __init__(
        self,
        h5_path: str,
        node_offsets: np.ndarray,
        edge_offsets: np.ndarray,
        has_labels: bool,
        node_cat_fields: List[str],
    ):
        self.h5_path = h5_path
        self.node_offsets = node_offsets
        self.edge_offsets = edge_offsets
        self.has_labels = has_labels
        self.node_cat_fields = node_cat_fields
        self._h5_file: Optional[h5py.File] = None
        self._pid: Optional[int] = None
    
    def _get_h5(self) -> h5py.File:
        """Get H5 file handle (reopens if process forked)."""
        pid = os.getpid()
        if self._h5_file is None or self._pid != pid:
            self.close()
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            self._pid = pid
        return self._h5_file
    
    def close(self):
        """Close H5 file."""
        if self._h5_file:
            try:
                self._h5_file.close()
            except:
                pass
            self._h5_file = None
    
    def __del__(self):
        self.close()
    
    def _compute_causal_mask(self, num_nodes: int, num_edges: int) -> torch.Tensor:
        """
        Compute causal mask for a single sample.
        
        For sequential edges (0→1, 1→2, ..., N-2→N-1):
        - Edge e connects node e to node e+1
        - When predicting edge e, nodes 0..e have happened
        
        Returns:
            causal_mask: (num_edges, num_nodes) boolean tensor
        """
        causal_mask = torch.zeros(num_edges, num_nodes, dtype=torch.bool)
        for e in range(num_edges):
            causal_mask[e, :e + 1] = True
        return causal_mask
    
    def __call__(self, indices: List[int]) -> Optional[Batch]:
        """
        Collate batch of samples with Time2Vec feature structure.
        
        Returns PyG Batch with:
        - node_observable_time: [total_nodes, 6] - time features for Time2Vec
        - node_observable_other: [total_nodes, 3] - other observable features
        - node_realized_time: [total_nodes, 6] - realized time features
        - node_realized_other: [total_nodes, realized_other_dim] - other realized
        - causal_masks: List of (num_edges_i, num_nodes_i) tensors
        - node_counts, edge_counts
        - All categorical indices
        - edge_index, edge_features
        - package_features
        - edge_labels (if available)
        """
        if not indices:
            return None
        
        f = self._get_h5()
        data_list = []
        causal_masks = []
        
        for idx in indices:
            n_start, n_end = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
            e_start, e_end = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
            num_nodes = n_end - n_start
            num_edges = e_end - e_start
            
            # Skip samples with no edges
            if num_edges == 0:
                continue
            
            # Read node features - Time2Vec structure
            node_obs_time = f['node_observable_time'][n_start:n_end]
            node_obs_other = f['node_observable_other'][n_start:n_end]
            node_real_time = f['node_realized_time'][n_start:n_end]
            node_real_other = f['node_realized_other'][n_start:n_end]
            
            # Read node categorical
            node_cat = {k: f['node_categorical'][k][n_start:n_end] for k in self.node_cat_fields}
            
            # Read edge features (adjust indices to local)
            edge_idx = f['edge_index'][:, e_start:e_end] - n_start
            edge_feat = f['edge_features'][e_start:e_end]
            
            # Read package features
            pkg_feat = f['package_features'][idx]
            
            # Compute causal mask
            causal_mask = self._compute_causal_mask(num_nodes, num_edges)
            causal_masks.append(causal_mask)
            
            # Build Data object with Time2Vec structure
            data = Data(
                # Node time features (for Time2Vec)
                node_observable_time=to_tensor(node_obs_time, torch.float32),
                node_realized_time=to_tensor(node_real_time, torch.float32),
                
                # Node other features
                node_observable_other=to_tensor(node_obs_other, torch.float32),
                node_realized_other=to_tensor(node_real_other, torch.float32),
                
                # Node categorical
                event_type_idx=to_tensor(node_cat['event_type'], torch.long),
                location_idx=to_tensor(node_cat['location'], torch.long),
                postal_idx=to_tensor(node_cat['postal'], torch.long),
                region_idx=to_tensor(node_cat['region'], torch.long),
                carrier_idx=to_tensor(node_cat['carrier'], torch.long),
                leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
                ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
                
                # Edge features
                edge_index=to_tensor(edge_idx, torch.long),
                edge_features=to_tensor(edge_feat, torch.float32),
                
                # Package features
                package_features=to_tensor(pkg_feat, torch.float32),
                source_postal_idx=torch.tensor([int(f['source_postal'][idx])], dtype=torch.long),
                dest_postal_idx=torch.tensor([int(f['dest_postal'][idx])], dtype=torch.long),
                
                # Metadata
                num_nodes=num_nodes,
            )
            
            # Labels
            if self.has_labels:
                data.edge_labels = to_tensor(f['edge_labels'][e_start:e_end], torch.float32)
                data.edge_labels_raw = to_tensor(f['edge_labels_raw'][e_start:e_end], torch.float32)
            
            data_list.append(data)
        
        if not data_list:
            return None
        
        # Batch using PyG
        batch = Batch.from_data_list(data_list)
        
        # Store causal masks as list
        batch.causal_masks = causal_masks
        
        # Store counts for indexing
        batch.node_counts = torch.tensor([d.num_nodes for d in data_list], dtype=torch.long)
        batch.edge_counts = torch.tensor([d.edge_index.shape[1] for d in data_list], dtype=torch.long)
        
        return batch