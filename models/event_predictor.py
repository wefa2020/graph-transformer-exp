"""
models/event_predictor.py - Event Time Predictor wrapper

Wraps CausalGraphTransformer with additional inference utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from models.graph_transformer import GraphTransformerLayer
from config import Config, ModelConfig


class EventTimePredictor(nn.Module):
    """
    Complete model for predicting next event time with ZERO feature leakage.
    
    Wraps CausalGraphTransformer and provides:
    - Standard forward pass for training
    - Inference utilities (predict, predict_next_event, etc.)
    - Checkpoint save/load functionality
    
    Use `from_config()` to create from Config object.
    Use `from_checkpoint()` to load from saved checkpoint.
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        embed_dim: int,
        output_dim: int,
        use_edge_features: bool,
    ):
        super().__init__()
        
        # Store for checkpoint saving
        self.vocab_sizes = vocab_sizes
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        
        # Initialize Graph Transformer
        self.graph_transformer = GraphTransformerLayer(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            embed_dim=embed_dim,
            output_dim=output_dim,
            use_edge_features=use_edge_features,
        )
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with zero feature leakage.
        
        Args:
            data: PyG Data batch from PackageLifecycleDataset
            
        Returns:
            predictions: [num_edges, output_dim] aligned with data.edge_labels
        """
        return self.graph_transformer(data)
    
    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        return self.forward(data)
    
    @torch.no_grad()
    def predict_next_event(self, data) -> torch.Tensor:
        """
        Predict time to next event for the last edge in each graph.
        Useful for inference on partial lifecycles.
        """
        self.eval()
        predictions = self.forward(data)
        
        if not hasattr(data, 'edge_counts'):
            return predictions
        
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
        return torch.zeros((0, self.output_dim), device=predictions.device)
    
    @torch.no_grad()
    def predict_all_transitions(self, data) -> List[Dict]:
        """Predict time for all transitions in each package lifecycle."""
        self.eval()
        predictions = self.forward(data)
        
        if not hasattr(data, 'edge_counts') or not hasattr(data, 'node_counts'):
            return [{'graph_idx': 0, 'predictions': predictions.cpu().numpy().flatten().tolist()}]
        
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
    
    @torch.no_grad()
    def predict_with_uncertainty(self, data, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty estimation using MC Dropout."""
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(n_samples):
            samples.append(self.forward(data))
        
        samples = torch.stack(samples, dim=0)
        self.eval()
        
        return {
            'mean': samples.mean(dim=0),
            'std': samples.std(dim=0),
            'samples': samples,
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        return self.graph_transformer.get_num_parameters()
    
    def get_config_dict(self) -> Dict:
        """Get model configuration as dict (for saving)."""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'embed_dim': self.embed_dim,
            'output_dim': self.output_dim,
            'use_edge_features': self.use_edge_features,
        }
    
    @classmethod
    def from_config(
        cls, 
        config: Config, 
        vocab_sizes: Dict[str, int], 
        feature_dims: Dict[str, int],
        device: torch.device = None,
    ) -> 'EventTimePredictor':
        """
        Create model from Config object.
        
        Args:
            config: Config object with model settings
            vocab_sizes: Dict mapping category name to vocabulary size
            feature_dims: Dict with feature dimensions
            device: Target device
        
        Returns:
            Initialized EventTimePredictor model
        """
        model = cls(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            embed_dim=config.model.embed_dim,
            output_dim=config.model.output_dim,
            use_edge_features=config.model.use_edge_features,
        )
        
        if device is not None:
            model.to(device)
        
        return model
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None) -> 'EventTimePredictor':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device
        
        Returns:
            Loaded EventTimePredictor model
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        vocab_sizes = checkpoint['vocab_sizes']
        feature_dims = checkpoint['feature_dims']
        
        # Get model config from checkpoint
        if 'model_config' in checkpoint:
            cfg = checkpoint['model_config']
        elif 'config' in checkpoint:
            cfg = checkpoint['config'].get('model', {})
        else:
            raise ValueError("Checkpoint missing model configuration")
        
        defaults = ModelConfig()
        
        model = cls(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=cfg.get('hidden_dim', defaults.hidden_dim),
            num_layers=cfg.get('num_layers', defaults.num_layers),
            num_heads=cfg.get('num_heads', defaults.num_heads),
            dropout=cfg.get('dropout', defaults.dropout),
            embed_dim=cfg.get('embed_dim', defaults.embed_dim),
            output_dim=cfg.get('output_dim', defaults.output_dim),
            use_edge_features=cfg.get('use_edge_features', defaults.use_edge_features),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
        
        return model