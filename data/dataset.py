"""
data/dataset.py - Dataset with S3 H5 cache support
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
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================

_worker_preprocessor = None


def _init_worker(preprocessor_bytes: bytes):
    global _worker_preprocessor
    _worker_preprocessor = pickle.loads(preprocessor_bytes)


def _process_item(args):
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
    return path is not None and str(path).startswith('s3://')


def parse_s3_path(s3_path: str) -> tuple:
    path = s3_path[5:] if s3_path.startswith('s3://') else s3_path
    parts = path.split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def s3_exists(s3_path: str) -> bool:
    try:
        import boto3
        from botocore.exceptions import ClientError
        bucket, key = parse_s3_path(s3_path)
        boto3.client('s3').head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def s3_download(s3_path: str, local_path: str, log_fn: Callable = None):
    import boto3
    bucket, key = parse_s3_path(s3_path)
    if log_fn:
        log_fn(f"  Downloading s3://{bucket}/{key}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client('s3').download_file(bucket, key, local_path)
    if log_fn:
        log_fn(f"  ✓ Downloaded {os.path.getsize(local_path) / 1e6:.1f} MB")


def s3_upload(local_path: str, s3_path: str, log_fn: Callable = None):
    import boto3
    bucket, key = parse_s3_path(s3_path)
    if log_fn:
        log_fn(f"  Uploading to s3://{bucket}/{key}")
    boto3.client('s3').upload_file(local_path, bucket, key)
    if log_fn:
        log_fn(f"  ✓ Uploaded {os.path.getsize(local_path) / 1e6:.1f} MB")


# =============================================================================
# DATASET
# =============================================================================

class PackageLifecycleDataset(Dataset):
    """Dataset that uses S3 for H5 cache storage."""
    
    _NODE_CAT_FIELDS = [
        'event_type', 'from_location', 'to_location', 'to_postal',
        'from_region', 'to_region', 'carrier', 'leg_type', 'ship_method'
    ]
    _LOOKAHEAD_CAT_FIELDS = [
        'next_event_type', 'next_location', 'next_postal', 'next_region',
        'next_carrier', 'next_leg_type', 'next_ship_method'
    ]
    _EDGE_CAT_FIELDS = [
        'from_location', 'to_location', 'to_postal',
        'from_region', 'to_region',
        'carrier_from', 'carrier_to',
        'ship_method_from', 'ship_method_to'
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
        self._log = log_fn or (lambda x: None)
        self._num_workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
        
        self._s3_path: str = h5_cache_path
        self._local_path: str = None
        self._node_offsets: np.ndarray = None
        self._edge_offsets: np.ndarray = None
        self._num_samples: int = 0
        self._has_labels: bool = False
        self._collator: Optional['H5BatchCollator'] = None
        
        self._log(f"PackageLifecycleDataset: {h5_cache_path}")
        
        if load_from_cache and h5_cache_path:
            if s3_exists(h5_cache_path):
                self._init_from_s3(h5_cache_path)
                return
            self._log(f"  Cache not found, creating new")
        
        if df is not None and preprocessor is not None:
            if not preprocessor.fitted:
                raise ValueError("Preprocessor must be fitted")
            cache = self._process_dataframe(df, preprocessor)
            if save_to_cache and h5_cache_path:
                self._save_to_s3(cache, h5_cache_path)
            self._init_from_s3(h5_cache_path)
        elif not load_from_cache:
            raise ValueError("Provide (df, preprocessor) or set load_from_cache=True")
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> int:
        return idx
    
    def __del__(self):
        self.close()
        if self._local_path and os.path.exists(self._local_path):
            try:
                os.unlink(self._local_path)
            except:
                pass
    
    def close(self):
        if self._collator:
            self._collator.close()
            self._collator = None
    
    def get_collate_fn(self) -> 'H5BatchCollator':
        if self._collator:
            self._collator.close()
        self._collator = H5BatchCollator(
            h5_path=self._local_path,
            node_offsets=self._node_offsets,
            edge_offsets=self._edge_offsets,
            has_labels=self._has_labels,
            node_cat_fields=self._NODE_CAT_FIELDS,
            lookahead_cat_fields=self._LOOKAHEAD_CAT_FIELDS,
            edge_cat_fields=self._EDGE_CAT_FIELDS,
        )
        return self._collator
    
    def _init_from_s3(self, s3_path: str):
        """Download from S3 and initialize."""
        self._s3_path = s3_path
        
        # Create temp file for local access
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        temp_file.close()
        
        # Download
        s3_download(s3_path, self._local_path, self._log)
        
        # Read metadata
        with h5py.File(self._local_path, 'r') as f:
            self._num_samples = int(f.attrs['num_samples'])
            self._has_labels = bool(f.attrs.get('has_labels', False))
            self._node_offsets = f['node_offsets'][:]
            self._edge_offsets = f['edge_offsets'][:]
        
        self._log(f"  Loaded {self._num_samples} samples")
    
    def _process_dataframe(self, df, preprocessor) -> List[Dict]:
        """Process DataFrame in parallel."""
        preprocessor_bytes = pickle.dumps(preprocessor)
        work_items = [(idx, row.to_dict()) for idx, (_, row) in enumerate(df.iterrows())]
        total = len(work_items)
        
        self._log(f"Processing {total} samples...")
        
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
                    results_dict[idx] = features
                elif len(errors) < 10:
                    errors.append((idx, error))
                if (i + 1) % max(1, total // 5) == 0:
                    self._log(f"  Processed {i + 1}/{total}")
        
        self._log(f"  Done: {len(results_dict)} samples")
        return [results_dict[i] for i in sorted(results_dict.keys())]
    
    def _save_to_s3(self, cache: List[Dict], s3_path: str):
        """Save H5 to S3."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        local_path = temp_file.name
        temp_file.close()
        
        try:
            self._write_h5(cache, local_path)
            s3_upload(local_path, s3_path, self._log)
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)
    
    def _write_h5(self, cache: List[Dict], path: str):
        """Write H5 file."""
        n_samples = len(cache)
        
        # Calculate offsets
        node_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        edge_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        
        for i, data in enumerate(cache):
            node_offsets[i + 1] = node_offsets[i] + data['num_nodes']
            edge_offsets[i + 1] = edge_offsets[i] + data['edge_index'].shape[1]
        
        total_nodes = int(node_offsets[-1])
        total_edges = int(edge_offsets[-1])
        
        sample0 = cache[0]
        node_cont_dim = sample0['node_continuous_features'].shape[1]
        edge_cont_dim = sample0['edge_continuous_features'].shape[1]
        has_labels = 'labels' in sample0
        
        self._log(f"  Writing H5: {total_nodes:,} nodes, {total_edges:,} edges")
        
        # Allocate
        node_continuous = np.zeros((total_nodes, node_cont_dim), dtype=np.float32)
        node_cat = {f: np.zeros(total_nodes, dtype=np.int32) for f in self._NODE_CAT_FIELDS}
        look_cat = {f: np.zeros(total_nodes, dtype=np.int32) for f in self._LOOKAHEAD_CAT_FIELDS}
        edge_index = np.zeros((2, total_edges), dtype=np.int64)
        edge_continuous = np.zeros((total_edges, edge_cont_dim), dtype=np.float32)
        edge_cat = {f: np.zeros(total_edges, dtype=np.int32) for f in self._EDGE_CAT_FIELDS}
        source_postal = np.zeros(n_samples, dtype=np.int32)
        dest_postal = np.zeros(n_samples, dtype=np.int32)
        
        if has_labels:
            labels = np.zeros(total_edges, dtype=np.float32)
            label_mask = np.zeros(total_nodes, dtype=np.uint8)
        
        # Fill
        for i, data in enumerate(cache):
            n_s, n_e = int(node_offsets[i]), int(node_offsets[i + 1])
            e_s, e_e = int(edge_offsets[i]), int(edge_offsets[i + 1])
            
            node_continuous[n_s:n_e] = data['node_continuous_features']
            for f in self._NODE_CAT_FIELDS:
                node_cat[f][n_s:n_e] = data['node_categorical_indices'][f]
            for f in self._LOOKAHEAD_CAT_FIELDS:
                look_cat[f][n_s:n_e] = data['lookahead_categorical_indices'][f]
            
            edge_index[:, e_s:e_e] = data['edge_index'] + n_s
            edge_continuous[e_s:e_e] = data['edge_continuous_features']
            for f in self._EDGE_CAT_FIELDS:
                edge_cat[f][e_s:e_e] = data['edge_categorical_indices'][f]
            
            source_postal[i] = data['package_categorical']['source_postal']
            dest_postal[i] = data['package_categorical']['dest_postal']
            
            if has_labels:
                labels[e_s:e_e] = data['labels'].flatten()
                label_mask[n_s:n_e] = data['label_mask'].astype(np.uint8)
        
        # Write
        with h5py.File(path, 'w') as f:
            f.attrs['num_samples'] = n_samples
            f.attrs['total_nodes'] = total_nodes
            f.attrs['total_edges'] = total_edges
            f.attrs['has_labels'] = has_labels
            
            f.create_dataset('node_offsets', data=node_offsets)
            f.create_dataset('edge_offsets', data=edge_offsets)
            f.create_dataset('source_postal', data=source_postal)
            f.create_dataset('dest_postal', data=dest_postal)
            f.create_dataset('node_continuous', data=node_continuous,
                           chunks=(min(10000, max(1, total_nodes)), node_cont_dim))
            
            grp = f.create_group('node_categorical')
            for k, v in node_cat.items():
                grp.create_dataset(k, data=v)
            
            grp = f.create_group('lookahead_categorical')
            for k, v in look_cat.items():
                grp.create_dataset(k, data=v)
            
            f.create_dataset('edge_index', data=edge_index)
            f.create_dataset('edge_continuous', data=edge_continuous,
                           chunks=(min(10000, max(1, total_edges)), edge_cont_dim))
            
            grp = f.create_group('edge_categorical')
            for k, v in edge_cat.items():
                grp.create_dataset(k, data=v)
            
            if has_labels:
                f.create_dataset('labels', data=labels)
                f.create_dataset('label_mask', data=label_mask)
        
        self._log(f"  ✓ H5 written: {os.path.getsize(path) / 1e6:.1f} MB")


# =============================================================================
# BATCH COLLATOR
# =============================================================================

class H5BatchCollator:
    """Collate function that reads batch from local H5 file."""
    
    def __init__(self, h5_path: str, node_offsets: np.ndarray, edge_offsets: np.ndarray,
                 has_labels: bool, node_cat_fields: List[str],
                 lookahead_cat_fields: List[str], edge_cat_fields: List[str]):
        self.h5_path = h5_path
        self.node_offsets = node_offsets
        self.edge_offsets = edge_offsets
        self.has_labels = has_labels
        self.node_cat_fields = node_cat_fields
        self.lookahead_cat_fields = lookahead_cat_fields
        self.edge_cat_fields = edge_cat_fields
        self._h5_file: Optional[h5py.File] = None
        self._pid: Optional[int] = None
    
    def _get_h5(self) -> h5py.File:
        pid = os.getpid()
        if self._h5_file is None or self._pid != pid:
            self.close()
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            self._pid = pid
        return self._h5_file
    
    def close(self):
        if self._h5_file:
            try:
                self._h5_file.close()
            except:
                pass
            self._h5_file = None
    
    def __del__(self):
        self.close()
    
    def __call__(self, indices: List[int]) -> Optional[Batch]:
        if not indices:
            return None
        
        f = self._get_h5()
        data_list = []
        
        for idx in indices:
            n_start, n_end = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
            e_start, e_end = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
            num_nodes = n_end - n_start
            
            node_cont = f['node_continuous'][n_start:n_end]
            edge_idx = f['edge_index'][:, e_start:e_end] - n_start
            edge_cont = f['edge_continuous'][e_start:e_end]
            
            node_cat = {k: f['node_categorical'][k][n_start:n_end] for k in self.node_cat_fields}
            look_cat = {k: f['lookahead_categorical'][k][n_start:n_end] for k in self.lookahead_cat_fields}
            edge_cat = {k: f['edge_categorical'][k][e_start:e_end] for k in self.edge_cat_fields}
            
            data = Data(
                node_continuous=to_tensor(node_cont, torch.float32),
                event_type_idx=to_tensor(node_cat['event_type'], torch.long),
                from_location_idx=to_tensor(node_cat['from_location'], torch.long),
                to_location_idx=to_tensor(node_cat['to_location'], torch.long),
                to_postal_idx=to_tensor(node_cat['to_postal'], torch.long),
                from_region_idx=to_tensor(node_cat['from_region'], torch.long),
                to_region_idx=to_tensor(node_cat['to_region'], torch.long),
                carrier_idx=to_tensor(node_cat['carrier'], torch.long),
                leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
                ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
                next_event_type_idx=to_tensor(look_cat['next_event_type'], torch.long),
                next_location_idx=to_tensor(look_cat['next_location'], torch.long),
                next_postal_idx=to_tensor(look_cat['next_postal'], torch.long),
                next_region_idx=to_tensor(look_cat['next_region'], torch.long),
                next_carrier_idx=to_tensor(look_cat['next_carrier'], torch.long),
                next_leg_type_idx=to_tensor(look_cat['next_leg_type'], torch.long),
                next_ship_method_idx=to_tensor(look_cat['next_ship_method'], torch.long),
                source_postal_idx=torch.tensor([int(f['source_postal'][idx])], dtype=torch.long),
                dest_postal_idx=torch.tensor([int(f['dest_postal'][idx])], dtype=torch.long),
                edge_index=to_tensor(edge_idx, torch.long),
                edge_continuous=to_tensor(edge_cont, torch.float32),
                edge_from_location_idx=to_tensor(edge_cat['from_location'], torch.long),
                edge_to_location_idx=to_tensor(edge_cat['to_location'], torch.long),
                edge_to_postal_idx=to_tensor(edge_cat['to_postal'], torch.long),
                edge_from_region_idx=to_tensor(edge_cat['from_region'], torch.long),
                edge_to_region_idx=to_tensor(edge_cat['to_region'], torch.long),
                edge_carrier_from_idx=to_tensor(edge_cat['carrier_from'], torch.long),
                edge_carrier_to_idx=to_tensor(edge_cat['carrier_to'], torch.long),
                edge_ship_method_from_idx=to_tensor(edge_cat['ship_method_from'], torch.long),
                edge_ship_method_to_idx=to_tensor(edge_cat['ship_method_to'], torch.long),
                num_nodes=num_nodes,
            )
            
            if self.has_labels:
                data.labels = to_tensor(f['labels'][e_start:e_end], torch.float32)
                data.label_mask = to_tensor(f['label_mask'][n_start:n_end].astype(bool), torch.bool)
            
            data_list.append(data)
        
        return Batch.from_data_list(data_list)