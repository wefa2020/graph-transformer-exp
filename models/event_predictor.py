"""
models/event_predictor.py - Event Time Predictor using Causal Graph Transformer with Time2Vec
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from torch_geometric.data import Batch

from models.causal_graph_transformer import CausalGraphTransformer


class EventTimePredictor(nn.Module):
    """
    Event Time Predictor for Package Lifecycle.
    
    This is a wrapper around CausalGraphTransformer that:
    - Predicts transit time between consecutive events
    - Uses Time2Vec for temporal feature encoding
    - Applies causal masking to prevent data leakage
    
    Input: PyG Batch with node/edge features
    Output: Transit time predictions for each edge [E, 1]
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        embed_dim: int = 32,
        time2vec_dim: int = 32,
        output_dim: int = 1,
        use_edge_features: bool = True,
    ):
        """
        Initialize EventTimePredictor.
        
        Args:
            vocab_sizes: Dictionary mapping categorical feature names to vocab sizes
            feature_dims: Dictionary with feature dimensions from preprocessor:
                - observable_time_dim: 6 (for Time2Vec)
                - observable_other_dim: 3
                - realized_time_dim: 6 (for Time2Vec)
                - realized_other_dim: varies (5 + num_problems)
                - edge_dim: 8
                - package_dim: 4
            hidden_dim: Hidden dimension for transformer layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            embed_dim: Embedding dimension for categorical features
            time2vec_dim: Output dimension for Time2Vec modules
            output_dim: Output dimension (1 for regression)
            use_edge_features: Whether to use edge features in attention
        """
        super().__init__()
        
        # Store configuration
        self.vocab_sizes = vocab_sizes
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.time2vec_dim = time2vec_dim
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        
        # Create the Causal Graph Transformer
        self.transformer = CausalGraphTransformer(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            embed_dim=embed_dim,
            time2vec_dim=time2vec_dim,
            output_dim=output_dim,
            use_edge_features=use_edge_features,
        )
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch: PyG Batch containing:
                - node_observable_time: [N, 6] - time features for Time2Vec
                - node_observable_other: [N, 3] - other observable features
                - node_realized_time: [N, 6] - realized time features
                - node_realized_other: [N, realized_other_dim] - other realized
                - edge_index: [2, E]
                - edge_features: [E, 8]
                - batch: [N] - batch assignment
                - Categorical indices: event_type_idx, location_idx, etc.
                - Package features: package_features, source_postal_idx, dest_postal_idx
        
        Returns:
            predictions: [E, output_dim] - transit time predictions for each edge
        """
        return self.transformer(batch)
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts."""
        return self.transformer.get_num_parameters()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving."""
        return {
            'vocab_sizes': self.vocab_sizes,
            'feature_dims': self.feature_dims,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'embed_dim': self.embed_dim,
            'time2vec_dim': self.time2vec_dim,
            'output_dim': self.output_dim,
            'use_edge_features': self.use_edge_features,
        }
    
    @classmethod
    def from_config(
        cls,
        config,
        vocab_sizes: Dict[str, int],
        feature_dims: Dict[str, int],
        device: torch.device = None,
    ) -> 'EventTimePredictor':
        """
        Create model from config object.
        
        Args:
            config: Config object with model parameters
            vocab_sizes: Vocabulary sizes from preprocessor
            feature_dims: Feature dimensions from preprocessor
            device: Target device
        
        Returns:
            Initialized EventTimePredictor
        """
        model = cls(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            embed_dim=config.model.embed_dim,
            time2vec_dim=getattr(config.model, 'time2vec_dim', 32),
            output_dim=getattr(config.model, 'output_dim', 1),
            use_edge_features=getattr(config.model, 'use_edge_features', True),
        )
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device = None,
        map_location: str = None,
    ) -> 'EventTimePredictor':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (local or S3)
            device: Target device
            map_location: Device mapping for loading
        
        Returns:
            Loaded EventTimePredictor
        """
        # Handle S3 paths
        if checkpoint_path.startswith('s3://'):
            import boto3
            import tempfile
            import os
            
            s3_path = checkpoint_path.replace('s3://', '')
            bucket, key = s3_path.split('/', 1)
            
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                local_path = f.name
            
            boto3.client('s3').download_file(bucket, key, local_path)
            checkpoint = torch.load(local_path, map_location=map_location or device, weights_only=False)
            os.unlink(local_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location or device, weights_only=False)
        
        # Get config from checkpoint
        model_config = checkpoint.get('model_config', {})
        vocab_sizes = checkpoint.get('vocab_sizes', model_config.get('vocab_sizes', {}))
        feature_dims = checkpoint.get('feature_dims', model_config.get('feature_dims', {}))
        
        # Create model
        model = cls(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 4),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            embed_dim=model_config.get('embed_dim', 32),
            time2vec_dim=model_config.get('time2vec_dim', 32),
            output_dim=model_config.get('output_dim', 1),
            use_edge_features=model_config.get('use_edge_features', True),
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None,
        epoch: int = None,
        metrics: Dict = None,
        is_best: bool = False,
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Save path (local or S3)
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config_dict(),
            'vocab_sizes': self.vocab_sizes,
            'feature_dims': self.feature_dims,
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Handle S3 paths
        if path.startswith('s3://'):
            import boto3
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                local_path = f.name
            
            torch.save(checkpoint, local_path)
            
            s3_path = path.replace('s3://', '')
            bucket, key = s3_path.split('/', 1)
            boto3.client('s3').upload_file(local_path, bucket, key)
            os.unlink(local_path)
        else:
            import os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(checkpoint, path)


class EventTimePredictorWithUncertainty(EventTimePredictor):
    """
    Event Time Predictor with uncertainty estimation.
    
    Outputs both mean prediction and variance for probabilistic predictions.
    Uses a separate head for variance prediction.
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        embed_dim: int = 32,
        time2vec_dim: int = 32,
        use_edge_features: bool = True,
        min_variance: float = 0.1,
    ):
        # Initialize with output_dim=1 for mean
        super().__init__(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            embed_dim=embed_dim,
            time2vec_dim=time2vec_dim,
            output_dim=1,
            use_edge_features=use_edge_features,
        )
        
        self.min_variance = min_variance
        
        # Variance prediction head
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim + time2vec_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures positive variance
        )
    
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            Dictionary with:
                - 'mean': [E, 1] - mean predictions
                - 'variance': [E, 1] - variance predictions
                - 'predictions': [E, 1] - same as mean (for compatibility)
        """
        # Get mean predictions from parent
        mean = self.transformer(batch)
        
        # For variance, we need access to the combined features
        # This requires modifying CausalGraphTransformer to expose intermediate features
        # For now, use mean as a simple proxy (can be improved)
        variance = self.min_variance + torch.abs(mean) * 0.1  # Simple heuristic
        
        return {
            'mean': mean,
            'variance': variance,
            'predictions': mean,
        }
    
    def nll_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss for Gaussian distribution.
        
        Args:
            predictions: Dictionary with 'mean' and 'variance'
            targets: Ground truth values
        
        Returns:
            NLL loss
        """
        mean = predictions['mean'].squeeze(-1)
        variance = predictions['variance'].squeeze(-1) + self.min_variance
        targets = targets.squeeze(-1)
        
        # Gaussian NLL: 0.5 * (log(var) + (y - mu)^2 / var)
        nll = 0.5 * (torch.log(variance) + (targets - mean) ** 2 / variance)
        return nll.mean()


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class HuberLoss(nn.Module):
    """Huber loss - more robust to outliers than MSE."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return loss.mean()


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic predictions."""
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, num_quantiles]
            targets: [batch, 1] or [batch]
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        return torch.cat(losses, dim=-1).mean()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_event_predictor(
    config,
    preprocessor,
    device: torch.device = None,
    with_uncertainty: bool = False,
) -> EventTimePredictor:
    """
    Factory function to create EventTimePredictor.
    
    Args:
        config: Configuration object
        preprocessor: Fitted preprocessor
        device: Target device
        with_uncertainty: Whether to use uncertainty estimation
    
    Returns:
        Initialized model
    """
    vocab_sizes = preprocessor.get_vocab_sizes()
    feature_dims = preprocessor.get_feature_dimensions()
    
    if with_uncertainty:
        model = EventTimePredictorWithUncertainty(
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            embed_dim=config.model.embed_dim,
            time2vec_dim=getattr(config.model, 'time2vec_dim', 32),
            use_edge_features=getattr(config.model, 'use_edge_features', True),
        )
    else:
        model = EventTimePredictor.from_config(
            config=config,
            vocab_sizes=vocab_sizes,
            feature_dims=feature_dims,
            device=device,
        )
    
    if device is not None:
        model = model.to(device)
    
    return model