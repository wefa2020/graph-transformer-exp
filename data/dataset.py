import torch
from torch.utils.data import Dataset
<<<<<<< HEAD
from torch_geometric.data import Data, Batch
from typing import List, Dict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import time

# Global worker state
_worker_preprocessor = None

def _init_worker(preprocessor):
    """Initialize preprocessor once per worker process"""
    global _worker_preprocessor
    _worker_preprocessor = preprocessor

def _process_single(record, return_labels):
    """Process single record using worker's preprocessor"""
    global _worker_preprocessor
    try:
        return _worker_preprocessor.process_lifecycle(record, return_labels=return_labels)
    except Exception as e:
        print(f"Error processing package {record.get('package_id', 'unknown')}: {e}")
        return None

class PackageLifecycleDataset(Dataset):
    """PyTorch dataset for package lifecycles - Optimized for CPU-intensive processing"""
    
    def __init__(
        self, 
        lifecycles_df,
        preprocessor,
        return_labels: bool = True,
        num_workers: int = None,
        progress_interval: int = 10000  # Print message every N items
    ):
        self.lifecycles_df = lifecycles_df
=======
from torch_geometric.data import Data
import pandas as pd
from typing import Optional

from data.data_preprocessor import PackageLifecyclePreprocessor


class PackageLifecycleDataset(Dataset):
    """PyTorch Geometric Dataset for package lifecycle data"""
    
    def __init__(self, df: pd.DataFrame, preprocessor: PackageLifecyclePreprocessor,
                 return_labels: bool = True):
        """
        Args:
            df: DataFrame with package lifecycle data
            preprocessor: Fitted PackageLifecyclePreprocessor
            return_labels: Whether to include labels (for training/validation)
        """
        self.df = df.reset_index(drop=True)
>>>>>>> 43a4a96 (large set 1)
        self.preprocessor = preprocessor
        self.return_labels = return_labels
        self.progress_interval = progress_interval
        
<<<<<<< HEAD
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        # Process all lifecycles in parallel
        self.processed_data = []
        
        # Convert to list of dicts
        df_records = lifecycles_df.to_dict('records')
        total_items = len(df_records)
        
        print(f"{'='*70}")
        print(f"Starting parallel processing:")
        print(f"  - Total lifecycles: {total_items:,}")
        print(f"  - Workers: {num_workers}")
        print(f"  - Progress updates every: {progress_interval:,} items")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(preprocessor,)
        ) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single, record, return_labels): idx 
                for idx, record in enumerate(df_records)
            }
            
            # Collect results with progress bar
            results = [None] * total_items
            completed_count = 0
            success_count = 0
            error_count = 0
            
            with tqdm(total=total_items, desc="Processing", unit="items") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results[idx] = result
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        print(f"\n⚠️  Error processing index {idx}: {e}")
                        error_count += 1
                    
                    pbar.update(1)
                    
                    # Print progress message every N items
                    if completed_count % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed
                        remaining = (total_items - completed_count) / rate if rate > 0 else 0
                        
                        print(f"\n{'─'*70}")
                        print(f"📊 Progress Update [{completed_count:,}/{total_items:,}]")
                        print(f"   ✅ Successful: {success_count:,}")
                        print(f"   ❌ Errors: {error_count:,}")
                        print(f"   ⚡ Speed: {rate:.1f} items/sec")
                        print(f"   ⏱️  Elapsed: {elapsed:.1f}s")
                        print(f"   ⏳ Est. remaining: {remaining:.1f}s")
                        print(f"{'─'*70}\n")
        
        # Filter out None values (failed processing)
        self.processed_data = [r for r in results if r is not None]
        
        # Final summary
        total_time = time.time() - start_time
        final_success = len(self.processed_data)
        final_errors = total_items - final_success
        
        print(f"\n{'='*70}")
        print(f"✨ Processing Complete!")
        print(f"{'='*70}")
        print(f"📈 Final Statistics:")
        print(f"   • Total items: {total_items:,}")
        print(f"   • Successfully processed: {final_success:,} ({final_success/total_items*100:.1f}%)")
        print(f"   • Failed: {final_errors:,} ({final_errors/total_items*100:.1f}%)")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Average speed: {total_items/total_time:.1f} items/sec")
        print(f"   • Time per item: {total_time/total_items*1000:.2f}ms")
        print(f"{'='*70}\n")
=======
        # Validate preprocessor is fitted
        if not preprocessor.fitted:
            raise ValueError("Preprocessor must be fitted before creating dataset")
>>>>>>> 43a4a96 (large set 1)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        
        try:
            processed = self.preprocessor.process_lifecycle(row, return_labels=self.return_labels)
        except Exception as e:
            print(f"Error processing package {row.get('package_id', idx)}: {e}")
            raise
        
        # Create PyG Data object
        data = Data(
            # === Node Features ===
            node_continuous=torch.tensor(
                processed['node_continuous_features'], dtype=torch.float32
            ),
            
            # Node categorical indices
            event_type_idx=torch.tensor(
                processed['node_categorical_indices']['event_type'], dtype=torch.long
            ),
            from_location_idx=torch.tensor(
                processed['node_categorical_indices']['from_location'], dtype=torch.long
            ),
            to_location_idx=torch.tensor(
                processed['node_categorical_indices']['to_location'], dtype=torch.long
            ),
            to_postal_idx=torch.tensor(
                processed['node_categorical_indices']['to_postal'], dtype=torch.long
            ),
            from_region_idx=torch.tensor(
                processed['node_categorical_indices']['from_region'], dtype=torch.long
            ),
            to_region_idx=torch.tensor(
                processed['node_categorical_indices']['to_region'], dtype=torch.long
            ),
            carrier_idx=torch.tensor(
                processed['node_categorical_indices']['carrier'], dtype=torch.long
            ),
            leg_type_idx=torch.tensor(
                processed['node_categorical_indices']['leg_type'], dtype=torch.long
            ),
            ship_method_idx=torch.tensor(
                processed['node_categorical_indices']['ship_method'], dtype=torch.long
            ),
            
            # Lookahead categorical indices
            next_event_type_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_event_type'], dtype=torch.long
            ),
            next_location_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_location'], dtype=torch.long
            ),
            next_postal_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_postal'], dtype=torch.long
            ),
            next_region_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_region'], dtype=torch.long
            ),
            next_carrier_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_carrier'], dtype=torch.long
            ),
            next_leg_type_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_leg_type'], dtype=torch.long
            ),
            next_ship_method_idx=torch.tensor(
                processed['lookahead_categorical_indices']['next_ship_method'], dtype=torch.long
            ),
            
            # Package categorical
            source_postal_idx=torch.tensor(
                [processed['package_categorical']['source_postal']], dtype=torch.long
            ),
            dest_postal_idx=torch.tensor(
                [processed['package_categorical']['dest_postal']], dtype=torch.long
            ),
            
            # === Edge Features ===
            edge_index=torch.tensor(processed['edge_index'], dtype=torch.long),
            edge_continuous=torch.tensor(
                processed['edge_continuous_features'], dtype=torch.float32
            ),
            
            # Edge categorical indices
            edge_from_location_idx=torch.tensor(
                processed['edge_categorical_indices']['from_location'], dtype=torch.long
            ),
            edge_to_location_idx=torch.tensor(
                processed['edge_categorical_indices']['to_location'], dtype=torch.long
            ),
            edge_to_postal_idx=torch.tensor(
                processed['edge_categorical_indices']['to_postal'], dtype=torch.long
            ),
            edge_from_region_idx=torch.tensor(
                processed['edge_categorical_indices']['from_region'], dtype=torch.long
            ),
            edge_to_region_idx=torch.tensor(
                processed['edge_categorical_indices']['to_region'], dtype=torch.long
            ),
            edge_carrier_from_idx=torch.tensor(
                processed['edge_categorical_indices']['carrier_from'], dtype=torch.long
            ),
            edge_carrier_to_idx=torch.tensor(
                processed['edge_categorical_indices']['carrier_to'], dtype=torch.long
            ),
            edge_ship_method_from_idx=torch.tensor(
                processed['edge_categorical_indices']['ship_method_from'], dtype=torch.long
            ),
            edge_ship_method_to_idx=torch.tensor(
                processed['edge_categorical_indices']['ship_method_to'], dtype=torch.long
            ),
            
            # Metadata
            num_nodes=processed['num_nodes'],
            package_id=processed['package_id']
        )
        
        if self.return_labels:
            data.labels = torch.tensor(processed['labels'], dtype=torch.float32)
            data.label_mask = torch.tensor(processed['label_mask'], dtype=torch.bool)
        
        return data