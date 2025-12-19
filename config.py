import torch
<<<<<<< HEAD
from dataclasses import dataclass, field
from typing import List, Optional
=======
from dataclasses import dataclass, field, asdict
from typing import List, Optional    
from typing import Dict

>>>>>>> 43a4a96 (large set 1)

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
    max_route_length = 20  # Adjust based on your data

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
<<<<<<< HEAD
    # Input dimensions
    node_feature_dim: int = 256
    edge_feature_dim: int = 64
    hidden_dim: int = 256
    
    # Graph Transformer
    num_layers: int = 40
    num_heads: int = 8
    dropout: float = 0.1
=======
    
    # === Input Dimensions ===
    node_feature_dim: int = 256          # Total node feature dim (auto-computed)
    edge_feature_dim: int = 64           # Total edge feature dim (auto-computed)
    hidden_dim: int = 256                # Hidden dimension for transformer
>>>>>>> 43a4a96 (large set 1)
    
    # === Continuous Feature Dimensions (from preprocessor) ===
    node_continuous_dim: int = 30        # Continuous node features
    edge_continuous_dim: int = 25        # Continuous edge features
    
    # === Embedding Dimensions ===
    embed_dim: int = 32                  # Embedding dimension for each categorical feature
    
    # === Graph Transformer ===
    num_layers: int = 40                 # Number of transformer layers
    num_heads: int = 8                   # Number of attention heads
    dropout: float = 0.1                 # Dropout rate
    
    # === Output ===
    output_dim: int = 1                  # Predicting time delta
    
    # === Architecture Choices ===
    use_edge_features: bool = True       # Whether to use edge features
    use_global_attention: bool = True    # Whether to use global attention pooling
    use_positional_encoding: bool = True # Whether to use positional encoding
    
    # === Categorical Feature Counts (set by preprocessor) ===
    num_node_categorical: int = 7        # event_type, location, from_location, region, carrier, leg_type, ship_method
    num_lookahead_categorical: int = 6   # next_event_type, next_location, next_region, next_carrier, next_leg_type, next_ship_method
    num_edge_categorical: int = 8        # from_location, to_location, from_region, to_region, carrier_from, carrier_to, ship_method_from, ship_method_to
    num_package_postal: int = 2          # source_postal, dest_postal
    
    def __post_init__(self):
        """Compute total feature dimensions after initialization"""
        # Node: continuous + node_cat + lookahead_cat + postal
        node_cat_dim = self.num_node_categorical * self.embed_dim
        lookahead_cat_dim = self.num_lookahead_categorical * self.embed_dim
        package_postal_dim = self.num_package_postal * self.embed_dim
        
        self.node_feature_dim = (
            self.node_continuous_dim +
            node_cat_dim +
            lookahead_cat_dim +
            package_postal_dim
        )
        
        # Edge: continuous + edge_cat
        edge_cat_dim = self.num_edge_categorical * self.embed_dim
        self.edge_feature_dim = self.edge_continuous_dim + edge_cat_dim
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        """Create from dictionary"""
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_fields})
    
    @classmethod
    def from_preprocessor(cls, preprocessor, **kwargs) -> 'ModelConfig':
        """
        Create config from fitted preprocessor
        
        Args:
            preprocessor: Fitted PackageLifecyclePreprocessor
            **kwargs: Override any config values (e.g., hidden_dim=512, num_layers=12)
        
        Returns:
            ModelConfig instance
        """
        feature_dims = preprocessor.get_feature_dimensions()
        
        return cls(
            node_continuous_dim=feature_dims['node_continuous_dim'],
            edge_continuous_dim=feature_dims['edge_continuous_dim'],
            num_node_categorical=feature_dims['num_node_categorical'],
            num_lookahead_categorical=feature_dims['num_lookahead_categorical'],
            num_edge_categorical=feature_dims['num_edge_categorical'],
            **kwargs
        )
        
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 128
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