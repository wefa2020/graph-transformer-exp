import torch
from dataclasses import dataclass, field
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
    event_types: List[str] = field(default_factory=lambda: ['INDUCT', 'EXIT', 'LINEHAUL', 'DELIVERY'])
    problem_types: List[str] = field(default_factory=lambda: [
        'NO_PROBLEM', 'DAMAGED_LABEL', 'WRONG_NODE', 'DAMAGED_PACKAGE_REPAIRABLE', 
        'DAMAGED_PACKAGE', 'DAMAGED_ITEM', 'CPT_EXPIRED', 'UNSUPPORTED_HAZMAT', 
        'PARTIAL_SHIPMENT', 'EMPTY_SHIPMENT', 'VAS_PRINT_LABEL', 'COMPLETE_SHIPMENT', 
        'CANCELLED', 'RESERVATION_EXPIRED', 'OVERWEIGHT', 'CRUSHED_BOX', 'REPACK', 
        'HOLE_IN_BOX', 'TAPE_ISSUE', 'NO_REPACK', 'ITEMS_MISSING_FROM_BIN', 
        'INCORRECT_DIMENSION', 'PACKAGE_OPEN', 'COMPLETE_REVERSE_SHIPMENT'
    ])
    
    # Features
    max_sequence_length: int = 20
    time_window_days: int = 30
    
    # Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    
    # Preprocessing
    normalize_time: bool = True
    add_positional_encoding: bool = True
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Input dimensions
    node_feature_dim: int = 256
    edge_feature_dim: int = 64
    hidden_dim: int = 256
    
    # Graph Transformer
    num_layers: int = 20
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
    batch_size: int = 64
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
    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment
    experiment_name: str = "package_event_prediction"
    seed: int = 42