import torch
from torch.utils.data import Dataset
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
        self.preprocessor = preprocessor
        self.return_labels = return_labels
        self.progress_interval = progress_interval
        
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
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        
        # Convert to PyG Data object
        x = torch.tensor(data['node_features'], dtype=torch.float)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(data['edge_features'], dtype=torch.float)
        
        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=data['num_nodes']
        )
        
        if self.return_labels:
            pyg_data.y = torch.tensor(data['labels'], dtype=torch.float)
            pyg_data.label_mask = torch.tensor(data['label_mask'], dtype=torch.bool)
        
        pyg_data.package_id = data['package_id']
        
        return pyg_data

def collate_fn(batch):
    """Custom collate function for batching"""
    return Batch.from_data_list(batch)