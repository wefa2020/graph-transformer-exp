import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict
import numpy as np

class PackageLifecycleDataset(Dataset):
    """PyTorch dataset for package lifecycles"""
    
    def __init__(
        self, 
        lifecycles_df,
        preprocessor,
        return_labels: bool = True
    ):
        self.lifecycles_df = lifecycles_df
        self.preprocessor = preprocessor
        self.return_labels = return_labels
        
        # Process all lifecycles
        self.processed_data = []
        for idx, row in lifecycles_df.iterrows():
            try:
                processed = preprocessor.process_lifecycle(
                    row.to_dict(),
                    return_labels=return_labels
                )
                self.processed_data.append(processed)
            except Exception as e:
                print(f"Error processing package {row['package_id']}: {e}")
                continue
    
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