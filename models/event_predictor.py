import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import ModelConfig
from models.graph_transformer import GraphTransformerWithEmbeddings


class EventTimePredictor(nn.Module):
    """Complete model for predicting next event time with ZERO feature leakage"""
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int], 
                 feature_dims: Dict = None):
        """
        Args:
            config: ModelConfig object with model architecture settings
            vocab_sizes: Dict mapping category name to vocabulary size
            feature_dims: Dict with 'observable_dim', 'realized_dim', 'edge_dim', 'package_dim'
        """
        super().__init__()
        
        self.config = config
        self.vocab_sizes = vocab_sizes
        self.feature_dims = feature_dims
        
        # Initialize Graph Transformer with embeddings
        self.graph_transformer = GraphTransformerWithEmbeddings(
            config, vocab_sizes, feature_dims
        )
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with zero feature leakage.
        
        For each edge (source → target):
        - Nodes 0 to source: use observable + realized features
        - Nodes source+1 to N-1: use observable + ZEROS (realized zeroed)
        
        Args:
            data: PyG Data batch from PackageLifecycleDataset
            
        Returns:
            predictions: Predicted time to next event for each edge [num_edges, 1]
        """
        predictions = self.graph_transformer(data)
        return predictions
    
    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        """
        Predict time to next event for all edges.
        
        Args:
            data: PyG Data batch
        
        Returns:
            predictions: Predicted times [num_edges, 1]
        """
        self.eval()
        predictions = self.forward(data)
        return predictions
    
    @torch.no_grad()
    def predict_next_event(self, data) -> torch.Tensor:
        """
        Predict time to next event for the last edge in each graph.
        Useful for inference on partial lifecycles.
        
        Args:
            data: PyG Data batch
            
        Returns:
            last_edge_preds: Predictions for last edge of each graph [batch_size, 1]
        """
        self.eval()
        
        predictions = self.forward(data)
        
        # Get prediction for last edge in each graph
        edge_counts = data.edge_counts.tolist()
        last_edge_preds = []
        
        edge_start = 0
        for count in edge_counts:
            if count > 0:
                last_pred = predictions[edge_start + count - 1]
                last_edge_preds.append(last_pred)
            edge_start += count
        
        if last_edge_preds:
            return torch.stack(last_edge_preds)
        return torch.zeros((0, 1), device=predictions.device)
    
    @torch.no_grad()
    def predict_all_transitions(self, data) -> List[Dict]:
        """
        Predict time for all transitions in each package lifecycle.
        
        Args:
            data: PyG Data batch
            
        Returns:
            List of dicts with predictions for each graph
        """
        self.eval()
        
        predictions = self.forward(data)
        
        edge_counts = data.edge_counts.tolist()
        node_counts = data.node_counts.tolist()
        
        results = []
        edge_start = 0
        
        for i, (n_edges, n_nodes) in enumerate(zip(edge_counts, node_counts)):
            graph_preds = predictions[edge_start:edge_start + n_edges].cpu().numpy()
            
            results.append({
                'graph_idx': i,
                'num_events': n_nodes,
                'num_transitions': n_edges,
                'predictions': graph_preds.flatten().tolist(),
            })
            
            edge_start += n_edges
        
        return results
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters"""
        return self.graph_transformer.get_num_parameters()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None) -> 'EventTimePredictor':
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device
        
        Returns:
            Loaded EventTimePredictor model
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct config
        config = ModelConfig.from_dict(checkpoint['model_config'])
        vocab_sizes = checkpoint['vocab_sizes']
        feature_dims = checkpoint.get('feature_dims', None)
        
        # Create model
        model = cls(config, vocab_sizes, feature_dims)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        return model
    
    @classmethod
    def from_preprocessor(cls, preprocessor, device: torch.device = None, 
                          **config_kwargs) -> 'EventTimePredictor':
        """
        Create model from fitted preprocessor
        
        Args:
            preprocessor: Fitted PackageLifecyclePreprocessor
            device: Target device
            **config_kwargs: Override config values (e.g., hidden_dim=512)
        
        Returns:
            Initialized EventTimePredictor model
        """
        config = ModelConfig.from_preprocessor(preprocessor, **config_kwargs)
        vocab_sizes = preprocessor.get_vocab_sizes()
        feature_dims = preprocessor.get_feature_dims()
        
        model = cls(config, vocab_sizes, feature_dims)
        
        if device is not None:
            model.to(device)
        
        return model