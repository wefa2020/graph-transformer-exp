import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import pickle
from tqdm.auto import tqdm
import multiprocessing as mp

from data.data_preprocessor import PackageLifecyclePreprocessor


# ============================================================================
# WORKER GLOBALS (initialized once per worker process)
# ============================================================================

_worker_preprocessor: Optional[PackageLifecyclePreprocessor] = None
_worker_return_labels: bool = True


def _init_worker(preprocessor_bytes: bytes, return_labels: bool):
    """Initialize worker process - called ONCE per worker."""
    global _worker_preprocessor, _worker_return_labels
    _worker_preprocessor = pickle.loads(preprocessor_bytes)
    _worker_return_labels = return_labels


def _process_single(args: Tuple[int, Dict]) -> Tuple[int, Optional[Dict], Optional[str]]:
    """
    Process single item in worker process.
    Uses global preprocessor (already loaded).
    
    Returns: (index, processed_dict_or_none, error_or_none)
    """
    global _worker_preprocessor, _worker_return_labels
    
    idx, row_dict = args
    try:
        processed = _worker_preprocessor.process_lifecycle(
            row_dict, 
            return_labels=_worker_return_labels
        )
        return (idx, processed, None)
    except Exception as e:
        return (idx, None, str(e))


# ============================================================================
# TENSOR CONVERSION
# ============================================================================

def _to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Fast numpy to tensor conversion."""
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


def _dict_to_data(processed: Dict[str, Any], return_labels: bool) -> Data:
    """Convert processed dict to PyG Data object."""
    node_cat = processed['node_categorical_indices']
    look_cat = processed['lookahead_categorical_indices']
    edge_cat = processed['edge_categorical_indices']
    pkg_cat = processed['package_categorical']
    
    data = Data(
        # Node features
        node_continuous=_to_tensor(processed['node_continuous_features'], torch.float32),
        event_type_idx=_to_tensor(node_cat['event_type'], torch.long),
        from_location_idx=_to_tensor(node_cat['from_location'], torch.long),
        to_location_idx=_to_tensor(node_cat['to_location'], torch.long),
        to_postal_idx=_to_tensor(node_cat['to_postal'], torch.long),
        from_region_idx=_to_tensor(node_cat['from_region'], torch.long),
        to_region_idx=_to_tensor(node_cat['to_region'], torch.long),
        carrier_idx=_to_tensor(node_cat['carrier'], torch.long),
        leg_type_idx=_to_tensor(node_cat['leg_type'], torch.long),
        ship_method_idx=_to_tensor(node_cat['ship_method'], torch.long),
        
        # Lookahead features
        next_event_type_idx=_to_tensor(look_cat['next_event_type'], torch.long),
        next_location_idx=_to_tensor(look_cat['next_location'], torch.long),
        next_postal_idx=_to_tensor(look_cat['next_postal'], torch.long),
        next_region_idx=_to_tensor(look_cat['next_region'], torch.long),
        next_carrier_idx=_to_tensor(look_cat['next_carrier'], torch.long),
        next_leg_type_idx=_to_tensor(look_cat['next_leg_type'], torch.long),
        next_ship_method_idx=_to_tensor(look_cat['next_ship_method'], torch.long),
        
        # Package categorical
        source_postal_idx=torch.tensor([pkg_cat['source_postal']], dtype=torch.long),
        dest_postal_idx=torch.tensor([pkg_cat['dest_postal']], dtype=torch.long),
        
        # Edge features
        edge_index=_to_tensor(processed['edge_index'], torch.long),
        edge_continuous=_to_tensor(processed['edge_continuous_features'], torch.float32),
        edge_from_location_idx=_to_tensor(edge_cat['from_location'], torch.long),
        edge_to_location_idx=_to_tensor(edge_cat['to_location'], torch.long),
        edge_to_postal_idx=_to_tensor(edge_cat['to_postal'], torch.long),
        edge_from_region_idx=_to_tensor(edge_cat['from_region'], torch.long),
        edge_to_region_idx=_to_tensor(edge_cat['to_region'], torch.long),
        edge_carrier_from_idx=_to_tensor(edge_cat['carrier_from'], torch.long),
        edge_carrier_to_idx=_to_tensor(edge_cat['carrier_to'], torch.long),
        edge_ship_method_from_idx=_to_tensor(edge_cat['ship_method_from'], torch.long),
        edge_ship_method_to_idx=_to_tensor(edge_cat['ship_method_to'], torch.long),
        
        # Metadata
        num_nodes=processed['num_nodes'],
        package_id=processed['package_id']
    )
    
    if return_labels:
        data.labels = _to_tensor(processed['labels'], torch.float32)
        data.label_mask = _to_tensor(processed['label_mask'], torch.bool)
    
    return data


# ============================================================================
# DATASET CLASS
# ============================================================================

class PackageLifecycleDataset(Dataset):
    """
    High-performance PyTorch Geometric Dataset.
    
    - Parallel preprocessing using multiprocessing Pool with initializer
    - Preprocessor loaded ONCE per worker (not per item)
    - Tensor conversion in main process
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        preprocessor: PackageLifecyclePreprocessor,
        return_labels: bool = True,
        num_workers: Optional[int] = None,
        chunksize: int = 1000,
    ):
        """
        Args:
            df: DataFrame with package lifecycle data
            preprocessor: Fitted PackageLifecyclePreprocessor
            return_labels: Whether to include labels
            num_workers: Number of parallel workers (default: cpu_count - 1)
            chunksize: Items per chunk for imap
        """
        if not preprocessor.fitted:
            raise ValueError("Preprocessor must be fitted before creating dataset")
        
        self.return_labels = return_labels
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.chunksize = chunksize
        
        # Convert DataFrame to list of dicts
        print("Converting DataFrame to records...")
        data_list: List[Dict] = df.to_dict('records')
        
        # Process in parallel
        self._data_cache: List[Data] = self._process_parallel(data_list, preprocessor)
    
    def _process_parallel(
        self, 
        data_list: List[Dict], 
        preprocessor: PackageLifecyclePreprocessor
    ) -> List[Data]:
        """Process all data in parallel."""
        
        # Serialize preprocessor ONCE
        print("Serializing preprocessor...")
        preprocessor_bytes = pickle.dumps(preprocessor)
        print(f"Preprocessor size: {len(preprocessor_bytes) / 1024 / 1024:.2f} MB")
        
        # Prepare indexed data
        indexed_data = [(i, row) for i, row in enumerate(data_list)]
        
        print(f"Processing {len(data_list)} items with {self.num_workers} workers "
              f"(chunksize={self.chunksize})...")
        
        # Results storage
        processed_dicts: List[Tuple[int, Dict]] = []
        errors: List[Tuple[int, str]] = []
        
        # Use 'fork' on Linux, 'spawn' elsewhere
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context('spawn')
        
        # Create pool with initializer
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(preprocessor_bytes, self.return_labels),
            maxtasksperchild=1000,
        ) as pool:
            
            results = pool.imap_unordered(
                _process_single,
                indexed_data,
                chunksize=self.chunksize
            )
            
            for idx, processed, error in tqdm(results, total=len(data_list), desc="Processing"):
                if processed is not None:
                    processed_dicts.append((idx, processed))
                else:
                    errors.append((idx, error))
        
        # Sort by index
        processed_dicts.sort(key=lambda x: x[0])
        
        # Report errors
        if errors:
            print(f"\n{len(errors)} items failed:")
            for idx, error in errors[:10]:
                print(f"  Index {idx}: {error}")
        
        # Convert to tensors in main process
        print("Converting to tensors...")
        data_cache = []
        for idx, processed in tqdm(processed_dicts, desc="Creating tensors"):
            data_cache.append(_dict_to_data(processed, self.return_labels))
        
        print(f"Successfully processed {len(data_cache)}/{len(data_list)} items")
        return data_cache
    
    def __len__(self) -> int:
        return len(self._data_cache)
    
    def __getitem__(self, idx: int) -> Data:
        return self._data_cache[idx]