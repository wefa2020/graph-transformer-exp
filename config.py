import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path

def _load_config_json() -> Dict[str, Any]:
    """Load config.json from the same directory as this module."""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

_CONFIG_JSON = _load_config_json()


@dataclass
class DataConfig:
    """Data paths."""
    cache_dir: str = "s3://graph-transformer-exp/cache"
    source_data: str = "s3://graph-transformer-exp/data/test.json"
    distance_file: Optional[str] = None
    num_workers: int = 8
    
    event_types: List[str] = field(default_factory=lambda: _CONFIG_JSON.get('event_types', []))
    problem_types: List[str] = field(default_factory=lambda: _CONFIG_JSON.get('problem_types', []))
    zip_codes: List[str] = field(default_factory=lambda: _CONFIG_JSON.get('zip_codes', []))
    
    @property
    def train_h5(self) -> str:
        return f"{self.cache_dir.rstrip('/')}/train.h5"
    
    @property
    def val_h5(self) -> str:
        return f"{self.cache_dir.rstrip('/')}/val.h5"
    
    @property
    def test_h5(self) -> str:
        return f"{self.cache_dir.rstrip('/')}/test.h5"
    
    @property
    def preprocessor_path(self) -> str:
        return f"{self.cache_dir.rstrip('/')}/preprocessor.pkl"


@dataclass
class ModelConfig:
    """Model architecture."""
    embed_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    output_dim: int = 1
    use_edge_features: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 200
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    patience: int = 15
    min_delta: float = 1e-4
    use_amp: bool = True
    checkpoint_frequency: int = 5
    seed: int = 42


@dataclass
class OutputConfig:
    """Output paths for model artifacts."""
    s3_output_dir: str = "s3://graph-transformer-exp/outputs"
    save_checkpoints: bool = True
    save_best_only: bool = False


@dataclass
class DistributedConfig:
    """Distributed training."""
    find_unused_parameters: bool = False


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    experiment_name: str = "causal_graph_transformer"
    
    @property
    def s3_experiment_dir(self) -> str:
        return f"{self.output.s3_output_dir.rstrip('/')}/{self.experiment_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'data': {
                'cache_dir': self.data.cache_dir,
                'source_data': self.data.source_data,
                'distance_file': self.data.distance_file,
                'num_workers': self.data.num_workers,
                'event_types': self.data.event_types,
                'problem_types': self.data.problem_types,
                'zip_codes': self.data.zip_codes,
            },
            'model': asdict(self.model),
            'training': asdict(self.training),
            'output': asdict(self.output),
            'distributed': asdict(self.distributed),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        data_dict = d.get('data', {})
        return cls(
            experiment_name=d.get('experiment_name', 'causal_graph_transformer'),
            data=DataConfig(
                cache_dir=data_dict.get('cache_dir', 's3://graph-transformer-exp/cache'),
                source_data=data_dict.get('source_data', ''),
                distance_file=data_dict.get('distance_file'),
                num_workers=data_dict.get('num_workers', 8),
                event_types=data_dict.get('event_types', _CONFIG_JSON.get('event_types', [])),
                problem_types=data_dict.get('problem_types', _CONFIG_JSON.get('problem_types', [])),
                zip_codes=data_dict.get('zip_codes', _CONFIG_JSON.get('zip_codes', [])),
            ),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            output=OutputConfig(**d.get('output', {})),
            distributed=DistributedConfig(**d.get('distributed', {})),
        )
    
    def save(self, path: str):
        if path.startswith('s3://'):
            import boto3
            path_clean = path.replace('s3://', '')
            bucket, key = path_clean.split('/', 1)
            json_bytes = json.dumps(self.to_dict(), indent=2).encode('utf-8')
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json_bytes)
        else:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        if path.startswith('s3://'):
            import boto3
            path_parts = path.replace('s3://', '').split('/', 1)
            bucket, key = path_parts[0], path_parts[1]
            response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return cls.from_dict(json.loads(content))
        else:
            with open(path, 'r') as f:
                return cls.from_dict(json.load(f))