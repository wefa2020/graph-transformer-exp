"""
data/dataset.py - Dataset with Shared Memory Support for Training
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
import pickle
import multiprocessing
import h5py
import os
import tempfile
import time
import ctypes


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
# SHARED MEMORY UTILITIES
# =============================================================================

def create_shared_array(shape: Tuple, dtype: np.dtype) -> np.ndarray:
    """Create a numpy array backed by shared memory."""
    size = int(np.prod(shape))
    
    # Map numpy dtype to ctypes
    dtype_map = {
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
        np.int32: ctypes.c_int32,
        np.int64: ctypes.c_int64,
    }
    
    # Handle numpy dtype objects
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        ctype = ctypes.c_float
    elif dtype == np.float64:
        ctype = ctypes.c_double
    elif dtype == np.int32:
        ctype = ctypes.c_int32
    elif dtype == np.int64:
        ctype = ctypes.c_int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Create shared memory array
    shared_base = mp.RawArray(ctype, size)
    shared_array = np.frombuffer(shared_base, dtype=dtype).reshape(shape)
    
    return shared_array


def copy_to_shared(arr: np.ndarray) -> np.ndarray:
    """Copy numpy array to shared memory."""
    shared = create_shared_array(arr.shape, arr.dtype)
    np.copyto(shared, arr)
    return shared


# =============================================================================
# WORKER FUNCTIONS FOR PARALLEL PREPROCESSING
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
# SHARED MEMORY DATA CONTAINER
# =============================================================================

class SharedMemoryData:
    """
    Container for dataset arrays in shared memory.
    
    All arrays are stored in shared memory (mp.RawArray) so they can be
    accessed by multiple DataLoader workers without copying.
    """
    
    NODE_CAT_FIELDS = [
        'event_type', 'location', 'postal', 'region',
        'carrier', 'leg_type', 'ship_method'
    ]
    
    def __init__(self):
        # Offsets
        self.node_offsets: Optional[np.ndarray] = None
        self.edge_offsets: Optional[np.ndarray] = None
        
        # Node features
        self.node_observable_time: Optional[np.ndarray] = None
        self.node_observable_other: Optional[np.ndarray] = None
        self.node_realized_time: Optional[np.ndarray] = None
        self.node_realized_other: Optional[np.ndarray] = None
        
        # Node categorical
        self.node_categorical: Dict[str, np.ndarray] = {}
        
        # Edge features
        self.edge_index: Optional[np.ndarray] = None
        self.edge_features: Optional[np.ndarray] = None
        
        # Package features
        self.package_features: Optional[np.ndarray] = None
        self.source_postal: Optional[np.ndarray] = None
        self.dest_postal: Optional[np.ndarray] = None
        
        # Labels
        self.edge_labels: Optional[np.ndarray] = None
        self.edge_labels_raw: Optional[np.ndarray] = None
        
        # Metadata
        self.num_samples: int = 0
        self.has_labels: bool = False
        self.obs_time_dim: int = 6
        self.obs_other_dim: int = 3
        self.real_time_dim: int = 6
        self.real_other_dim: int = 11
        self.edge_dim: int = 8
        self.package_dim: int = 4
    
    @classmethod
    def from_h5_file(cls, h5_path: str, log_fn: Callable = None) -> 'SharedMemoryData':
        """
        Load H5 file into shared memory.
        
        This is the key method for training - loads entire dataset into RAM
        for fast multi-worker access.
        """
        log = log_fn or (lambda x: None)
        data = cls()
        
        log(f"  Loading H5 into shared memory: {h5_path}")
        start_time = time.time()
        
        with h5py.File(h5_path, 'r') as f:
            # Read metadata
            data.num_samples = int(f.attrs['num_samples'])
            data.has_labels = bool(f.attrs.get('has_labels', False))
            data.obs_time_dim = int(f.attrs.get('obs_time_dim', 6))
            data.obs_other_dim = int(f.attrs.get('obs_other_dim', 3))
            data.real_time_dim = int(f.attrs.get('real_time_dim', 6))
            data.real_other_dim = int(f.attrs.get('real_other_dim', 11))
            data.edge_dim = int(f.attrs.get('edge_dim', 8))
            data.package_dim = int(f.attrs.get('package_dim', 4))
            
            total_nodes = int(f.attrs.get('total_nodes', f['node_offsets'][-1]))
            total_edges = int(f.attrs.get('total_edges', f['edge_offsets'][-1]))
            
            log(f"    Samples: {data.num_samples:,}, Nodes: {total_nodes:,}, Edges: {total_edges:,}")
            
            # Load offsets into shared memory
            log(f"    Loading offsets...")
            data.node_offsets = copy_to_shared(f['node_offsets'][:].astype(np.int64))
            data.edge_offsets = copy_to_shared(f['edge_offsets'][:].astype(np.int64))
            
            # Load node features into shared memory
            log(f"    Loading node features...")
            data.node_observable_time = copy_to_shared(f['node_observable_time'][:].astype(np.float32))
            data.node_observable_other = copy_to_shared(f['node_observable_other'][:].astype(np.float32))
            data.node_realized_time = copy_to_shared(f['node_realized_time'][:].astype(np.float32))
            data.node_realized_other = copy_to_shared(f['node_realized_other'][:].astype(np.float32))
            
            # Load node categorical into shared memory
            log(f"    Loading categorical features...")
            for field in cls.NODE_CAT_FIELDS:
                if field in f['node_categorical']:
                    data.node_categorical[field] = copy_to_shared(
                        f['node_categorical'][field][:].astype(np.int64)
                    )
            
            # Load edge features into shared memory
            log(f"    Loading edge features...")
            data.edge_index = copy_to_shared(f['edge_index'][:].astype(np.int64))
            data.edge_features = copy_to_shared(f['edge_features'][:].astype(np.float32))
            
            # Load package features into shared memory
            log(f"    Loading package features...")
            data.package_features = copy_to_shared(f['package_features'][:].astype(np.float32))
            data.source_postal = copy_to_shared(f['source_postal'][:].astype(np.int64))
            data.dest_postal = copy_to_shared(f['dest_postal'][:].astype(np.int64))
            
            # Load labels into shared memory
            if data.has_labels:
                log(f"    Loading labels...")
                data.edge_labels = copy_to_shared(f['edge_labels'][:].astype(np.float32))
                data.edge_labels_raw = copy_to_shared(f['edge_labels_raw'][:].astype(np.float32))
        
        elapsed = time.time() - start_time
        mem_mb = data.estimate_memory_mb()
        log(f"  ✓ Loaded into shared memory: {mem_mb:.1f} MB in {elapsed:.1f}s")
        
        return data
    
    def estimate_memory_mb(self) -> float:
        """Estimate total memory usage in MB."""
        total_bytes = 0
        
        for arr in [
            self.node_offsets, self.edge_offsets,
            self.node_observable_time, self.node_observable_other,
            self.node_realized_time, self.node_realized_other,
            self.edge_index, self.edge_features,
            self.package_features, self.source_postal, self.dest_postal,
            self.edge_labels, self.edge_labels_raw,
        ]:
            if arr is not None:
                total_bytes += arr.nbytes
        
        for arr in self.node_categorical.values():
            if arr is not None:
                total_bytes += arr.nbytes
        
        return total_bytes / (1024 * 1024)


# =============================================================================
# DATASET
# =============================================================================

class PackageLifecycleDataset(Dataset):
    """
    Dataset for Causal Graph Transformer.
    
    Two modes of operation:
    
    1. PREPROCESSING MODE (creating H5):
       - Provide df and preprocessor
       - Processes data and saves to H5 file
       - No shared memory used
       
    2. TRAINING MODE (loading H5):
       - Set load_from_cache=True
       - Optionally enable use_shared_memory=True for fast multi-worker access
       - Loads entire dataset into shared memory for training
    
    Example (Preprocessing):
        dataset = PackageLifecycleDataset(
            df=df,
            preprocessor=preprocessor,
            h5_cache_path="data/train.h5",
            save_to_cache=True,
        )
    
    Example (Training with shared memory):
        dataset = PackageLifecycleDataset(
            h5_cache_path="data/train.h5",
            load_from_cache=True,
            use_shared_memory=True,  # Load into RAM for fast access
        )
    """
    
    _NODE_CAT_FIELDS = SharedMemoryData.NODE_CAT_FIELDS
    
    def __init__(
        self,
        df=None,
        preprocessor=None,
        h5_cache_path: str = None,
        load_from_cache: bool = False,
        save_to_cache: bool = True,
        use_shared_memory: bool = False,
        num_workers: Optional[int] = None,
        log_fn: Callable = None,
        progress_callback: Callable[[int, int], None] = None
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with package data (for preprocessing)
            preprocessor: Fitted PackageLifecyclePreprocessor (for preprocessing)
            h5_cache_path: Path to H5 cache (local or S3)
            load_from_cache: If True, load from existing H5 cache
            save_to_cache: If True, save processed data to H5 cache
            use_shared_memory: If True, load H5 into shared memory (for training)
            num_workers: Number of parallel workers for preprocessing
            log_fn: Logging function
            progress_callback: Callback function(processed, total) for progress
        """
        self._log = log_fn or (lambda x: None)
        self._progress_callback = progress_callback
        self._num_workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
        
        # Storage mode flags
        self._use_shared_memory = use_shared_memory
        self._is_preprocessing = (df is not None and preprocessor is not None)
        
        # Paths
        self._s3_path: Optional[str] = None
        self._local_path: Optional[str] = None
        self._temp_file: bool = False
        
        # Shared memory data (only used in training mode)
        self._shared_data: Optional[SharedMemoryData] = None
        
        # H5 file handle (only used in H5 mode without shared memory)
        self._h5_file: Optional[h5py.File] = None
        
        # Offsets (used in both modes)
        self._node_offsets: Optional[np.ndarray] = None
        self._edge_offsets: Optional[np.ndarray] = None
        
        # Metadata
        self._num_samples: int = 0
        self._has_labels: bool = False
        self._collator = None
        
        # Feature dimensions
        self._obs_time_dim: int = 6
        self._obs_other_dim: int = 3
        self._real_time_dim: int = 6
        self._real_other_dim: int = 11
        self._edge_dim: int = 8
        self._package_dim: int = 4
        
        # Determine mode and initialize
        is_s3 = is_s3_path(h5_cache_path) if h5_cache_path else False
        
        if self._is_preprocessing:
            # =========================================================
            # PREPROCESSING MODE: Create H5 from DataFrame
            # =========================================================
            self._log(f"PackageLifecycleDataset: PREPROCESSING MODE")
            self._log(f"  Output: {h5_cache_path}")
            
            if not preprocessor.fitted:
                raise ValueError("Preprocessor must be fitted")
            
            # Process DataFrame to feature dicts
            cache = self._process_dataframe(df, preprocessor)
            
            if save_to_cache and h5_cache_path:
                if is_s3:
                    # Write to temp file, upload to S3
                    self._save_and_upload_to_s3(cache, h5_cache_path)
                else:
                    # Write directly to local path
                    os.makedirs(os.path.dirname(h5_cache_path) or '.', exist_ok=True)
                    self._write_h5(cache, h5_cache_path)
                    self._local_path = h5_cache_path
                    self._read_h5_metadata()
            else:
                # No cache path - write to temp file
                self._init_from_cache_list(cache)
        
        elif load_from_cache and h5_cache_path:
            # =========================================================
            # TRAINING MODE: Load from H5 cache
            # =========================================================
            mode_str = "SHARED MEMORY" if use_shared_memory else "H5 FILE"
            self._log(f"PackageLifecycleDataset: TRAINING MODE ({mode_str})")
            self._log(f"  Source: {h5_cache_path}")
            
            if is_s3:
                if s3_exists(h5_cache_path):
                    self._init_from_s3(h5_cache_path)
                else:
                    raise FileNotFoundError(f"S3 cache not found: {h5_cache_path}")
            else:
                if os.path.exists(h5_cache_path):
                    self._init_from_local(h5_cache_path)
                else:
                    raise FileNotFoundError(f"Local cache not found: {h5_cache_path}")
        else:
            raise ValueError(
                "Either provide (df, preprocessor) for preprocessing, "
                "or set load_from_cache=True with h5_cache_path"
            )
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> int:
        return idx
    
    def __del__(self):
        self.close()
    
    def close(self):
        """Clean up resources."""
        if self._collator:
            self._collator.close()
            self._collator = None
        
        if self._h5_file:
            try:
                self._h5_file.close()
            except:
                pass
            self._h5_file = None
        
        # Clean up temp file
        if self._temp_file and self._local_path and os.path.exists(self._local_path):
            try:
                os.unlink(self._local_path)
            except:
                pass
    
    def get_collate_fn(self):
        """Get collate function for DataLoader."""
        if self._collator:
            self._collator.close()
        
        if self._use_shared_memory and self._shared_data is not None:
            # Fast shared memory collator
            self._collator = SharedMemoryCollator(
                shared_data=self._shared_data,
                node_cat_fields=self._NODE_CAT_FIELDS,
            )
        else:
            # H5 file-based collator
            self._collator = H5FileCollator(
                h5_path=self._local_path,
                node_offsets=self._node_offsets,
                edge_offsets=self._edge_offsets,
                has_labels=self._has_labels,
                node_cat_fields=self._NODE_CAT_FIELDS,
            )
        
        return self._collator
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions."""
        return {
            'obs_time_dim': self._obs_time_dim,
            'obs_other_dim': self._obs_other_dim,
            'real_time_dim': self._real_time_dim,
            'real_other_dim': self._real_other_dim,
            'edge_dim': self._edge_dim,
            'package_dim': self._package_dim,
        }
    
    def get_shared_data(self) -> Optional[SharedMemoryData]:
        """Get shared memory data container."""
        return self._shared_data
    
    def is_shared_memory_mode(self) -> bool:
        """Check if dataset is using shared memory."""
        return self._shared_data is not None
    
    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================
    
    def _init_from_s3(self, s3_path: str):
        """Initialize from S3 path."""
        self._s3_path = s3_path
        
        # Download to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        s3_download(s3_path, self._local_path, self._log)
        
        # Load data based on mode
        self._load_data()
    
    def _init_from_local(self, local_path: str):
        """Initialize from local path."""
        self._local_path = local_path
        self._temp_file = False
        
        # Load data based on mode
        self._load_data()
    
    def _init_from_cache_list(self, cache: List[Dict]):
        """Initialize from processed cache list (temp file)."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        self._write_h5(cache, self._local_path)
        self._read_h5_metadata()
    
    def _save_and_upload_to_s3(self, cache: List[Dict], s3_path: str):
        """Save cache to temp file and upload to S3."""
        self._s3_path = s3_path
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self._local_path = temp_file.name
        self._temp_file = True
        temp_file.close()
        
        self._write_h5(cache, self._local_path)
        s3_upload(self._local_path, s3_path, self._log)
        self._read_h5_metadata()
    
    def _load_data(self):
        """Load data from H5 file (into shared memory if enabled)."""
        if self._use_shared_memory:
            # Load entire H5 into shared memory
            self._shared_data = SharedMemoryData.from_h5_file(self._local_path, self._log)
            
            # Copy metadata
            self._num_samples = self._shared_data.num_samples
            self._has_labels = self._shared_data.has_labels
            self._obs_time_dim = self._shared_data.obs_time_dim
            self._obs_other_dim = self._shared_data.obs_other_dim
            self._real_time_dim = self._shared_data.real_time_dim
            self._real_other_dim = self._shared_data.real_other_dim
            self._edge_dim = self._shared_data.edge_dim
            self._package_dim = self._shared_data.package_dim
            
            # Keep offsets reference
            self._node_offsets = self._shared_data.node_offsets
            self._edge_offsets = self._shared_data.edge_offsets
            
            # Delete temp file after loading into memory
            if self._temp_file and self._local_path and os.path.exists(self._local_path):
                try:
                    os.unlink(self._local_path)
                    self._log("  Deleted temp H5 file (data now in shared memory)")
                    self._local_path = None
                    self._temp_file = False
                except:
                    pass
        else:
            # Just read metadata, keep H5 file for lazy loading
            self._read_h5_metadata()
    
    def _read_h5_metadata(self):
        """Read metadata from H5 file."""
        with h5py.File(self._local_path, 'r') as f:
            self._num_samples = int(f.attrs['num_samples'])
            self._has_labels = bool(f.attrs.get('has_labels', False))
            self._obs_time_dim = int(f.attrs.get('obs_time_dim', 6))
            self._obs_other_dim = int(f.attrs.get('obs_other_dim', 3))
            self._real_time_dim = int(f.attrs.get('real_time_dim', 6))
            self._real_other_dim = int(f.attrs.get('real_other_dim', 11))
            self._edge_dim = int(f.attrs.get('edge_dim', 8))
            self._package_dim = int(f.attrs.get('package_dim', 4))
            self._node_offsets = f['node_offsets'][:].astype(np.int64)
            self._edge_offsets = f['edge_offsets'][:].astype(np.int64)
        
        self._log(f"  Loaded metadata: {self._num_samples:,} samples")
    
    def _report_progress(self, processed: int, total: int):
        """Report progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(processed, total)
            except:
                pass
    
    # =========================================================================
    # PREPROCESSING METHODS
    # =========================================================================
    
    def _process_dataframe(self, df, preprocessor) -> List[Dict]:
        """Process DataFrame with parallel workers."""
        preprocessor_bytes = pickle.dumps(preprocessor)
        work_items = [(idx, row.to_dict()) for idx, (_, row) in enumerate(df.iterrows())]
        total = len(work_items)
        
        self._log(f"  Processing {total:,} samples with {self._num_workers} workers...")
        
        results_dict = {}
        errors = []
        processed_count = 0
        valid_count = 0
        skipped_count = 0
        
        start_time = time.time()
        last_log_time = start_time
        log_interval = 5.0
        progress_interval = max(1, total // 100)
        
        try:
            ctx = multiprocessing.get_context('fork')
        except ValueError:
            ctx = multiprocessing.get_context('spawn')
        
        self._report_progress(0, total)
        
        with ctx.Pool(self._num_workers, initializer=_init_worker,
                      initargs=(preprocessor_bytes,), maxtasksperchild=1000) as pool:
            
            for idx, features, error in pool.imap_unordered(_process_item, work_items, chunksize=500):
                processed_count += 1
                
                if features is not None:
                    if features['num_nodes'] >= 2:
                        results_dict[idx] = features
                        valid_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
                    if len(errors) < 10:
                        errors.append((idx, error))
                
                if processed_count % progress_interval == 0 or processed_count == total:
                    self._report_progress(processed_count, total)
                
                current_time = time.time()
                if current_time - last_log_time >= log_interval or processed_count == total:
                    elapsed = current_time - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    pct = 100 * processed_count / total
                    eta = (total - processed_count) / rate if rate > 0 else 0
                    
                    self._log(
                        f"    Progress: {processed_count:,}/{total:,} ({pct:.1f}%) | "
                        f"Valid: {valid_count:,} | Rate: {rate:.0f}/s | ETA: {eta:.0f}s"
                    )
                    last_log_time = current_time
        
        self._report_progress(total, total)
        
        elapsed = time.time() - start_time
        self._log(f"  Processing complete in {elapsed:.1f}s")
        self._log(f"    Valid: {valid_count:,}, Skipped: {skipped_count:,}")
        
        if errors:
            self._log(f"  Errors ({len(errors)}):")
            for err_idx, err_msg in errors[:5]:
                self._log(f"    [{err_idx}] {err_msg}")
        
        return [results_dict[i] for i in sorted(results_dict.keys())]
    
    def _write_h5(self, cache: List[Dict], path: str):
        """Write processed data to H5 file."""
        n_samples = len(cache)
        if n_samples == 0:
            raise ValueError("No samples to write")
        
        sample0 = cache[0]
        
        # Calculate offsets
        node_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        edge_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        
        for i, data in enumerate(cache):
            node_offsets[i + 1] = node_offsets[i] + data['num_nodes']
            edge_offsets[i + 1] = edge_offsets[i] + data['edge_index'].shape[1]
        
        total_nodes = int(node_offsets[-1])
        total_edges = int(edge_offsets[-1])
        
        # Get dimensions
        obs_time_dim = sample0['node_observable_time'].shape[1]
        obs_other_dim = sample0['node_observable_other'].shape[1]
        real_time_dim = sample0['node_realized_time'].shape[1]
        real_other_dim = sample0['node_realized_other'].shape[1]
        edge_dim = sample0['edge_features'].shape[1]
        
        pkg_feat = sample0.get('package_features')
        package_dim = pkg_feat.shape[0] if pkg_feat.ndim == 1 else pkg_feat.shape[1]
        
        has_labels = 'labels' in sample0
        
        self._log(f"  Writing H5: {n_samples:,} samples, {total_nodes:,} nodes, {total_edges:,} edges")
        
        # Allocate arrays
        node_obs_time = np.zeros((total_nodes, obs_time_dim), dtype=np.float32)
        node_obs_other = np.zeros((total_nodes, obs_other_dim), dtype=np.float32)
        node_real_time = np.zeros((total_nodes, real_time_dim), dtype=np.float32)
        node_real_other = np.zeros((total_nodes, real_other_dim), dtype=np.float32)
        
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
            
            node_obs_time[n_s:n_e] = data['node_observable_time']
            node_obs_other[n_s:n_e] = data['node_observable_other']
            node_real_time[n_s:n_e] = data['node_realized_time']
            node_real_other[n_s:n_e] = data['node_realized_other']
            
            cat_data = data.get('node_categorical_indices', {})
            for f in self._NODE_CAT_FIELDS:
                if f in cat_data:
                    node_cat[f][n_s:n_e] = cat_data[f]
            
            edge_index[:, e_s:e_e] = data['edge_index'] + n_s
            edge_features[e_s:e_e] = data['edge_features']
            
            pkg = data['package_features']
            package_features[i] = pkg if pkg.ndim == 1 else pkg.flatten()[:package_dim]
            
            pkg_cat = data.get('package_categorical', {})
            source_postal[i] = pkg_cat.get('source_postal', 0)
            dest_postal[i] = pkg_cat.get('dest_postal', 0)
            
            if has_labels:
                lbl = data['labels']
                edge_labels[e_s:e_e] = lbl.flatten() if lbl.ndim > 1 else lbl
                lbl_raw = data.get('labels_raw', data['labels'])
                edge_labels_raw[e_s:e_e] = lbl_raw.flatten() if lbl_raw.ndim > 1 else lbl_raw
        
        # Write H5
        with h5py.File(path, 'w') as f:
            f.attrs['num_samples'] = n_samples
            f.attrs['total_nodes'] = total_nodes
            f.attrs['total_edges'] = total_edges
            f.attrs['has_labels'] = has_labels
            f.attrs['obs_time_dim'] = obs_time_dim
            f.attrs['obs_other_dim'] = obs_other_dim
            f.attrs['real_time_dim'] = real_time_dim
            f.attrs['real_other_dim'] = real_other_dim
            f.attrs['edge_dim'] = edge_dim
            f.attrs['package_dim'] = package_dim
            
            f.create_dataset('node_offsets', data=node_offsets)
            f.create_dataset('edge_offsets', data=edge_offsets)
            
            chunk_size = min(10000, max(1, total_nodes))
            f.create_dataset('node_observable_time', data=node_obs_time,
                           chunks=(chunk_size, obs_time_dim))
            f.create_dataset('node_observable_other', data=node_obs_other,
                           chunks=(chunk_size, obs_other_dim))
            f.create_dataset('node_realized_time', data=node_real_time,
                           chunks=(chunk_size, real_time_dim))
            f.create_dataset('node_realized_other', data=node_real_other,
                           chunks=(chunk_size, real_other_dim))
            
            grp = f.create_group('node_categorical')
            for k, v in node_cat.items():
                grp.create_dataset(k, data=v)
            
            f.create_dataset('edge_index', data=edge_index)
            edge_chunk = min(10000, max(1, total_edges))
            f.create_dataset('edge_features', data=edge_features,
                           chunks=(edge_chunk, edge_dim))
            
            f.create_dataset('package_features', data=package_features)
            f.create_dataset('source_postal', data=source_postal)
            f.create_dataset('dest_postal', data=dest_postal)
            
            if has_labels:
                f.create_dataset('edge_labels', data=edge_labels)
                f.create_dataset('edge_labels_raw', data=edge_labels_raw)
        
        file_size_mb = os.path.getsize(path) / 1e6
        self._log(f"  ✓ H5 written: {file_size_mb:.1f} MB")


# =============================================================================
# SHARED MEMORY COLLATOR (FAST - FOR TRAINING)
# =============================================================================

class SharedMemoryCollator:
    """
    Fast collate function that reads from shared memory.
    
    No I/O operations - direct memory access.
    Safe for multi-process DataLoader workers (fork inherits shared memory).
    """
    
    def __init__(
        self,
        shared_data: SharedMemoryData,
        node_cat_fields: List[str],
    ):
        self.data = shared_data
        self.node_cat_fields = node_cat_fields
    
    def close(self):
        """No cleanup needed for shared memory."""
        pass
    
    def __call__(self, indices: List[int]) -> Optional[Batch]:
        if not indices:
            return None
        
        data_list = []
        d = self.data  # Local reference for speed
        
        for idx in indices:
            n_start = int(d.node_offsets[idx])
            n_end = int(d.node_offsets[idx + 1])
            e_start = int(d.edge_offsets[idx])
            e_end = int(d.edge_offsets[idx + 1])
            
            num_nodes = n_end - n_start
            num_edges = e_end - e_start
            
            if num_edges == 0:
                continue
            
            # Direct memory slicing (no copy until tensor creation)
            node_obs_time = d.node_observable_time[n_start:n_end]
            node_obs_other = d.node_observable_other[n_start:n_end]
            node_real_time = d.node_realized_time[n_start:n_end]
            node_real_other = d.node_realized_other[n_start:n_end]
            
            # Edge features - adjust indices
            edge_idx = d.edge_index[:, e_start:e_end] - n_start
            edge_feat = d.edge_features[e_start:e_end]
            
            # Package features
            pkg_feat = d.package_features[idx:idx+1]
            
            # Create PyG Data object
            data = Data(
                node_observable_time=to_tensor(node_obs_time, torch.float32),
                node_observable_other=to_tensor(node_obs_other, torch.float32),
                node_realized_time=to_tensor(node_real_time, torch.float32),
                node_realized_other=to_tensor(node_real_other, torch.float32),
                
                event_type_idx=to_tensor(d.node_categorical['event_type'][n_start:n_end], torch.long),
                location_idx=to_tensor(d.node_categorical['location'][n_start:n_end], torch.long),
                postal_idx=to_tensor(d.node_categorical['postal'][n_start:n_end], torch.long),
                region_idx=to_tensor(d.node_categorical['region'][n_start:n_end], torch.long),
                carrier_idx=to_tensor(d.node_categorical['carrier'][n_start:n_end], torch.long),
                leg_type_idx=to_tensor(d.node_categorical['leg_type'][n_start:n_end], torch.long),
                ship_method_idx=to_tensor(d.node_categorical['ship_method'][n_start:n_end], torch.long),
                
                edge_index=to_tensor(edge_idx, torch.long),
                edge_features=to_tensor(edge_feat, torch.float32),
                
                package_features=to_tensor(pkg_feat, torch.float32),
                source_postal_idx=torch.tensor([int(d.source_postal[idx])], dtype=torch.long),
                dest_postal_idx=torch.tensor([int(d.dest_postal[idx])], dtype=torch.long),
                
                num_nodes=num_nodes,
            )
            
            # Add labels if available
            if d.has_labels and d.edge_labels is not None:
                data.edge_labels = to_tensor(d.edge_labels[e_start:e_end], torch.float32)
                data.edge_labels_raw = to_tensor(d.edge_labels_raw[e_start:e_end], torch.float32)
            
            data_list.append(data)
        
        if not data_list:
            return None
        
        # Batch with PyG
        batch = Batch.from_data_list(data_list)
        batch.node_counts = torch.tensor([d.num_nodes for d in data_list], dtype=torch.long)
        batch.edge_counts = torch.tensor([d.edge_index.shape[1] for d in data_list], dtype=torch.long)
        
        return batch


# =============================================================================
# H5 FILE COLLATOR (FALLBACK - FOR PREPROCESSING OR LOW MEMORY)
# =============================================================================

class H5FileCollator:
    """
    Collate function that reads from H5 file.
    
    Use this when:
    - Memory is limited (can't fit dataset in RAM)
    - During preprocessing (before shared memory is needed)
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
        """Get H5 file handle (re-open if in new process)."""
        pid = os.getpid()
        if self._h5_file is None or self._pid != pid:
            self.close()
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            self._pid = pid
        return self._h5_file
    
    def close(self):
        """Close H5 file handle."""
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
            num_edges = e_end - e_start
            
            if num_edges == 0:
                continue
            
            # Read from H5 file
            node_obs_time = f['node_observable_time'][n_start:n_end]
            node_obs_other = f['node_observable_other'][n_start:n_end]
            node_real_time = f['node_realized_time'][n_start:n_end]
            node_real_other = f['node_realized_other'][n_start:n_end]
            
            node_cat = {k: f['node_categorical'][k][n_start:n_end] for k in self.node_cat_fields}
            
            edge_idx = f['edge_index'][:, e_start:e_end] - n_start
            edge_feat = f['edge_features'][e_start:e_end]
            pkg_feat = f['package_features'][idx:idx+1]
            
            data = Data(
                node_observable_time=to_tensor(node_obs_time, torch.float32),
                node_observable_other=to_tensor(node_obs_other, torch.float32),
                node_realized_time=to_tensor(node_real_time, torch.float32),
                node_realized_other=to_tensor(node_real_other, torch.float32),
                
                event_type_idx=to_tensor(node_cat['event_type'], torch.long),
                location_idx=to_tensor(node_cat['location'], torch.long),
                postal_idx=to_tensor(node_cat['postal'], torch.long),
                region_idx=to_tensor(node_cat['region'], torch.long),
                carrier_idx=to_tensor(node_cat['carrier'], torch.long),
                leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
                ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
                
                edge_index=to_tensor(edge_idx, torch.long),
                edge_features=to_tensor(edge_feat, torch.float32),
                
                package_features=to_tensor(pkg_feat, torch.float32),
                source_postal_idx=torch.tensor([int(f['source_postal'][idx])], dtype=torch.long),
                dest_postal_idx=torch.tensor([int(f['dest_postal'][idx])], dtype=torch.long),
                
                num_nodes=num_nodes,
            )
            
            if self.has_labels:
                data.edge_labels = to_tensor(f['edge_labels'][e_start:e_end], torch.float32)
                data.edge_labels_raw = to_tensor(f['edge_labels_raw'][e_start:e_end], torch.float32)
            
            data_list.append(data)
        
        if not data_list:
            return None
        
        batch = Batch.from_data_list(data_list)
        batch.node_counts = torch.tensor([d.num_nodes for d in data_list], dtype=torch.long)
        batch.edge_counts = torch.tensor([d.edge_index.shape[1] for d in data_list], dtype=torch.long)
        
        return batch