import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class NeptuneConfig:
    """Neptune database configuration"""
    endpoint: str = "swa-shipgraph-neptune-instance-prod-us-east-1.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
    use_iam: bool = False
    region: str = "us-east-1"

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Event types in order
    event_types: List[str] = None
    
    # Features
    max_sequence_length: int = 20
    time_window_days: int = 30
    
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Preprocessing
    normalize_time: bool = True
    add_positional_encoding: bool = True
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = ['INDUCT', 'EXIT', 'LINEHAUL', 'DELIVERY']

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Input dimensions
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    hidden_dim: int = 128
    
    # Graph Transformer
    num_layers: int = 15
    num_heads: int = 8
    dropout: float = 0.1
    
    # Output
    output_dim: int = 1  # Predicting time delta
    
    # Architecture choices
    use_edge_features: bool = True
    use_global_attention: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 30
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 5

@dataclass
class Config:
    """Main configuration"""
    neptune: NeptuneConfig = NeptuneConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Experiment
    experiment_name: str = "package_event_prediction"
    seed: int = 42