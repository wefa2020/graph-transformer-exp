#!/usr/bin/env python3
"""
train_distributed.py - Distributed training for Causal Graph Transformer
with Shared Memory support for maximum GPU utilization
"""

import os
import sys
import json
import signal
import gc
import time
from datetime import timedelta
from contextlib import nullcontext
import logging
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config


# =============================================================================
# MULTIPROCESSING SETUP (MUST BE FIRST)
# =============================================================================

def setup_multiprocessing():
    """
    Set multiprocessing start method to 'fork' for shared memory inheritance.
    Must be called before any multiprocessing operations.
    """
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass


setup_multiprocessing()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class RankLogger:
    def __init__(self, rank: int):
        self.rank = rank
        self._logger = logging.getLogger(__name__)
    
    def info(self, msg):
        if self.rank == 0:
            self._logger.info(msg)
    
    def warning(self, msg):
        if self.rank == 0:
            self._logger.warning(msg)
    
    def error(self, msg):
        self._logger.error(f"[Rank {self.rank}] {msg}")
    
    def debug(self, msg):
        if self.rank == 0:
            self._logger.debug(msg)


def get_logger(rank: int) -> RankLogger:
    return RankLogger(rank)


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_distributed():
    """Setup distributed training environment."""
    env_vars = {
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "FI_PROVIDER": "efa",
        "FI_EFA_FORK_SAFE": "1",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if world_size > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl', init_method='env://',
                world_size=world_size, rank=rank,
                timeout=timedelta(days=7),
                device_id=torch.device(f'cuda:{local_rank}')
            )
        except TypeError:
            dist.init_process_group(
                backend='nccl', init_method='env://',
                world_size=world_size, rank=rank,
                timeout=timedelta(days=7)
            )
    
    return rank, local_rank, world_size


def sync_barrier(device: torch.device):
    """Synchronize all distributed processes."""
    if dist.is_initialized():
        dist.barrier()
        dummy = torch.zeros(1, device=device)
        dist.all_reduce(dummy)
        del dummy
        torch.cuda.synchronize()


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility."""
    actual_seed = seed + rank
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    np.random.seed(actual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class GracefulShutdown:
    """Handle graceful shutdown on signals."""
    
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.warning(f"Received signal {signum}, shutting down...")
        self.should_stop = True


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def get_optimal_num_workers(world_size: int, local_rank: int) -> int:
    """Calculate optimal number of DataLoader workers."""
    cpu_count = os.cpu_count() or 8
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    workers_per_gpu = max(4, cpu_count // gpus_per_node)
    return min(workers_per_gpu, 16)


# =============================================================================
# S3 UPLOAD UTILITIES
# =============================================================================

class S3ModelUploader:
    """Handle uploading models to S3."""
    
    def __init__(
        self, 
        bucket: str, 
        job_name: str, 
        log: RankLogger,
        prefix: str = "output"
    ):
        self.bucket = bucket
        self.job_name = job_name
        self.prefix = prefix
        self.log = log
        self._s3_client = None
    
    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client('s3')
        return self._s3_client
    
    @property
    def best_model_s3_path(self) -> str:
        """Get the S3 path for best model."""
        return f"s3://{self.bucket}/{self.prefix}/{self.job_name}/best_model/"
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Upload a single file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path within bucket)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            return True
        except Exception as e:
            self.log.error(f"Failed to upload {local_path} to s3://{self.bucket}/{s3_key}: {e}")
            return False
    
    def upload_best_model(self, local_model_path: str, additional_files: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload best model and optional additional files to S3.
        
        Args:
            local_model_path: Path to the local model checkpoint file
            additional_files: Dict of {local_path: s3_filename} for additional files
            
        Returns:
            True if all uploads successful, False otherwise
        """
        if not os.path.exists(local_model_path):
            self.log.error(f"Model file not found: {local_model_path}")
            return False
        
        # Upload main model file
        model_filename = os.path.basename(local_model_path)
        s3_key = f"{self.prefix}/{self.job_name}/best_model/{model_filename}"
        
        self.log.info(f"  Uploading best model to {self.best_model_s3_path}")
        
        success = self.upload_file(local_model_path, s3_key)
        
        if success:
            self.log.info(f"  ✓ Model uploaded to s3://{self.bucket}/{s3_key}")
        
        # Upload additional files if provided
        if additional_files and success:
            for local_path, s3_filename in additional_files.items():
                if os.path.exists(local_path):
                    add_s3_key = f"{self.prefix}/{self.job_name}/best_model/{s3_filename}"
                    if self.upload_file(local_path, add_s3_key):
                        self.log.info(f"  ✓ Uploaded {s3_filename}")
                    else:
                        success = False
        
        return success
    
    def upload_directory(self, local_dir: str, s3_subpath: str = "best_model") -> bool:
        """
        Upload all files in a directory to S3.
        
        Args:
            local_dir: Local directory path
            s3_subpath: Subpath within the job output folder
            
        Returns:
            True if all uploads successful, False otherwise
        """
        if not os.path.isdir(local_dir):
            self.log.error(f"Directory not found: {local_dir}")
            return False
        
        success = True
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                s3_key = f"{self.prefix}/{self.job_name}/{s3_subpath}/{filename}"
                if not self.upload_file(local_path, s3_key):
                    success = False
        
        return success


def get_s3_uploader(config: Config, log: RankLogger) -> Optional[S3ModelUploader]:
    """
    Create S3 uploader from config and environment variables.
    
    Args:
        config: Configuration object
        log: Logger instance
        
    Returns:
        S3ModelUploader instance or None if configuration is incomplete
    """
    # Get bucket from config or environment
    bucket = getattr(config.data, 's3_output_bucket', None)
    if bucket is None:
        bucket = os.environ.get('SM_HP_S3_OUTPUT_BUCKET', None)
    if bucket is None:
        # Try to extract from output path if available
        output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '')
        if output_path.startswith('s3://'):
            bucket = output_path.replace('s3://', '').split('/')[0]
    
    # Get job name from environment
    job_name = os.environ.get('SM_TRAINING_JOB_NAME', None)
    if job_name is None:
        job_name = os.environ.get('TRAINING_JOB_NAME', None)
    if job_name is None:
        # Fallback to experiment name with timestamp
        job_name = f"{config.experiment_name}_{int(time.time())}"
    
    # Get prefix from config or use default
    prefix = getattr(config.data, 's3_output_prefix', 'output')
    
    if bucket is None:
        log.warning("S3 bucket not configured. Best model will only be saved locally.")
        log.warning("Set SM_HP_S3_OUTPUT_BUCKET or config.data.s3_output_bucket to enable S3 upload.")
        return None
    
    log.info(f"S3 Upload configured:")
    log.info(f"  Bucket: {bucket}")
    log.info(f"  Job Name: {job_name}")
    log.info(f"  Output Path: s3://{bucket}/{prefix}/{job_name}/best_model/")
    
    return S3ModelUploader(bucket=bucket, job_name=job_name, log=log, prefix=prefix)


# =============================================================================
# DATA LOADING WITH SHARED MEMORY
# =============================================================================

def load_data_shared_memory(
    config: Config, 
    rank: int, 
    local_rank: int, 
    world_size: int, 
    log: RankLogger
) -> Tuple:
    """Load datasets into shared memory for fast multi-worker access."""
    import boto3
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        log.info("=" * 60)
        log.info("LOADING DATA INTO SHARED MEMORY")
        log.info("=" * 60)
        log.info(f"  Train: {config.data.train_h5}")
        log.info(f"  Val:   {config.data.val_h5}")
        log.info(f"  Test:  {config.data.test_h5}")
    
    sync_barrier(device)
    
    log_fn = log.info if rank == 0 else None
    
    start_time = time.time()
    
    train_dataset = PackageLifecycleDataset(
        h5_cache_path=config.data.train_h5,
        load_from_cache=True,
        save_to_cache=False,
        use_shared_memory=True,
        log_fn=log_fn
    )
    
    val_dataset = PackageLifecycleDataset(
        h5_cache_path=config.data.val_h5,
        load_from_cache=True,
        save_to_cache=False,
        use_shared_memory=True,
        log_fn=log_fn
    )
    
    test_dataset = PackageLifecycleDataset(
        h5_cache_path=config.data.test_h5,
        load_from_cache=True,
        save_to_cache=False,
        use_shared_memory=True,
        log_fn=log_fn
    )
    
    load_time = time.time() - start_time
    
    if rank == 0:
        total_memory_mb = 0
        for ds in [train_dataset, val_dataset, test_dataset]:
            if ds._shared_data is not None:
                total_memory_mb += ds._shared_data.estimate_memory_mb()
        
        log.info(f"  Total shared memory usage: {total_memory_mb:.1f} MB")
        log.info(f"  Data loading time: {load_time:.1f}s")
    
    # Load preprocessor
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    local_preprocessor = os.path.join(model_dir, 'preprocessor.pkl')
    
    if not os.path.exists(local_preprocessor):
        s3_path = config.data.preprocessor_path.replace('s3://', '')
        bucket, key = s3_path.split('/', 1)
        os.makedirs(os.path.dirname(local_preprocessor), exist_ok=True)
        boto3.client('s3').download_file(bucket, key, local_preprocessor)
    
    preprocessor = PackageLifecyclePreprocessor.load(local_preprocessor)
    
    sync_barrier(device)
    
    if rank == 0:
        log.info(f"  Samples: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")
        log.info("=" * 60)
    
    return train_dataset, val_dataset, test_dataset, preprocessor


def create_dataloaders(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    config: Config, 
    rank: int, 
    local_rank: int,
    world_size: int
) -> Tuple:
    """Create DataLoaders optimized for shared memory."""
    log = get_logger(rank)
    
    batch_size = config.training.batch_size
    num_workers = get_optimal_num_workers(world_size, local_rank)
    
    if hasattr(config.data, 'num_workers') and config.data.num_workers > 0:
        num_workers = config.data.num_workers
    
    log.info(f"  DataLoader workers per GPU: {num_workers}")
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context='fork'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=val_dataset.get_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='fork'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=test_dataset.get_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='fork'
    )
    
    log.info(f"  DataLoaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    log.info(f"  Batch size: {batch_size}, Workers: {num_workers}, Prefetch: 4")
    
    return train_loader, val_loader, test_loader, train_sampler


# =============================================================================
# MODEL
# =============================================================================

def create_model(
    preprocessor, 
    config: Config, 
    device: torch.device, 
    local_rank: int, 
    world_size: int, 
    rank: int
) -> Tuple:
    """Create the Causal Graph Transformer model."""
    from models.event_predictor import EventTimePredictor
    
    log = get_logger(rank)
    vocab_sizes = preprocessor.get_vocab_sizes()
    feature_dims = preprocessor.get_feature_dimensions()
    
    if rank == 0:
        log.info(f"Feature dims: {feature_dims}")
        log.info(f"Vocab sizes: {vocab_sizes}")
    
    model = EventTimePredictor.from_config(
        config=config,
        vocab_sizes=vocab_sizes,
        feature_dims=feature_dims,
        device=device,
    )
    
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=config.distributed.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True
        )
        log.info("Model wrapped with DDP (static_graph=True)")
    
    return model, vocab_sizes, feature_dims


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    
    if len(preds) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0}
    
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    mask = targets != 0
    mape = float(np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100) if mask.sum() > 0 else 0.0
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def reduce_metrics(metrics: Dict, device: torch.device, world_size: int) -> Dict:
    """Reduce metrics across distributed workers."""
    if world_size <= 1:
        return metrics
    
    reduced = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(float(v), device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = t.item() / world_size
        else:
            reduced[k] = v
    return reduced


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model, optimizer, scheduler, epoch: int, metrics: Dict,
    vocab_sizes: Dict, feature_dims: Dict, config: Config,
    local_path: str, is_best: bool = False
):
    """Save checkpoint locally."""
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'vocab_sizes': vocab_sizes,
        'feature_dims': feature_dims,
        'model_config': model_to_save.get_config_dict(),
        'config': config.to_dict(),
        'metrics': metrics,
        'is_best': is_best
    }
    
    temp_path = local_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, local_path)


def save_and_upload_best_model(
    model, optimizer, scheduler, epoch: int, metrics: Dict,
    vocab_sizes: Dict, feature_dims: Dict, config: Config,
    local_path: str, s3_uploader: Optional[S3ModelUploader],
    preprocessor_path: Optional[str] = None,
    log: Optional[RankLogger] = None
) -> bool:
    """
    Save checkpoint locally and upload to S3.
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Training metrics
        vocab_sizes: Vocabulary sizes dict
        feature_dims: Feature dimensions dict
        config: Config object
        local_path: Local path to save the model
        s3_uploader: S3ModelUploader instance (optional)
        preprocessor_path: Path to preprocessor file to upload alongside model (optional)
        log: Logger instance (optional)
        
    Returns:
        True if save (and upload if configured) was successful
    """
    # Save locally first
    save_checkpoint(
        model, optimizer, scheduler, epoch, metrics,
        vocab_sizes, feature_dims, config, local_path, is_best=True
    )
    
    # Upload to S3 if uploader is configured
    if s3_uploader is not None:
        additional_files = {}
        
        # Include preprocessor if available
        if preprocessor_path and os.path.exists(preprocessor_path):
            additional_files[preprocessor_path] = 'preprocessor.pkl'
        
        # Include config
        config_path = local_path.replace('.pt', '_config.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            additional_files[config_path] = 'config.json'
        except Exception as e:
            if log:
                log.warning(f"Failed to save config: {e}")
        
        # Include metrics
        metrics_path = local_path.replace('.pt', '_metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
                }, f, indent=2)
            additional_files[metrics_path] = 'metrics.json'
        except Exception as e:
            if log:
                log.warning(f"Failed to save metrics: {e}")
        
        # Upload to S3
        success = s3_uploader.upload_best_model(local_path, additional_files)
        
        # Cleanup temporary files
        for temp_file in [config_path, metrics_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        return success
    
    return True


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model, 
    loader, 
    optimizer, 
    criterion, 
    device: torch.device, 
    preprocessor,
    config: Config,
    rank: int, 
    world_size: int, 
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    num_optimizer_steps = 0
    skipped_batches = 0
    
    grad_accum = config.training.gradient_accumulation_steps
    max_grad_norm = config.training.max_grad_norm
    use_amp = config.training.use_amp
    
    is_ddp = world_size > 1 and hasattr(model, 'no_sync')
    num_batches = len(loader)
    
    if num_batches == 0:
        return {'loss': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0, 'num_optimizer_steps': 0}
    
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_samples = 0
    
    batch_times = []
    data_times = []
    compute_times = []
    
    data_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
        data_time = time.time() - data_start
        data_times.append(data_time)
        
        compute_start = time.time()
        
        batch = batch.to(device, non_blocking=True)
        
        should_sync = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches
        ctx = model.no_sync() if (is_ddp and not should_sync) else nullcontext()
        
        with ctx:
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(batch)
                edge_labels = batch.edge_labels
                
                preds_flat = preds.squeeze(-1) if preds.dim() > 1 else preds
                labels_flat = edge_labels.squeeze(-1) if edge_labels.dim() > 1 else edge_labels
                
                if preds_flat.numel() == 0:
                    skipped_batches += 1
                    data_start = time.time()
                    continue
                
                min_len = min(len(preds_flat), len(labels_flat))
                preds_flat, labels_flat = preds_flat[:min_len], labels_flat[:min_len]
                
                if torch.isnan(preds_flat).any() or torch.isinf(preds_flat).any():
                    skipped_batches += 1
                    data_start = time.time()
                    continue
                
                loss = criterion(preds_flat.float(), labels_flat.float())
                
                if torch.isnan(loss) or torch.isinf(loss):
                    skipped_batches += 1
                    data_start = time.time()
                    continue
                
                scaled_loss = loss / grad_accum
            
            scaled_loss.backward()
        
        bs = preds_flat.size(0)
        accumulated_loss += loss.item() * bs
        accumulated_samples += bs
        
        all_preds.append(preprocessor.inverse_transform_time(
            preds_flat.detach().float().cpu().numpy()
        ))
        all_targets.append(preprocessor.inverse_transform_time(
            labels_flat.float().cpu().numpy()
        ))
        
        if should_sync:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()
                num_optimizer_steps += 1
            
            optimizer.zero_grad(set_to_none=True)
            total_loss += accumulated_loss
            total_samples += accumulated_samples
            accumulated_loss = 0.0
            accumulated_samples = 0
        
        compute_time = time.time() - compute_start
        compute_times.append(compute_time)
        batch_times.append(data_time + compute_time)
        
        # Log progress
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            avg_batch_time = np.mean(batch_times[-100:])
            avg_data_time = np.mean(data_times[-100:])
            avg_compute_time = np.mean(compute_times[-100:])
            throughput = bs / avg_batch_time if avg_batch_time > 0 else 0
            
            logging.info(
                f"  Batch {batch_idx + 1}/{num_batches} | "
                f"Throughput: {throughput:.0f} samples/s | "
                f"Data: {avg_data_time*1000:.1f}ms | "
                f"Compute: {avg_compute_time*1000:.1f}ms"
            )
        
        data_start = time.time()
    
    if accumulated_samples > 0:
        total_loss += accumulated_loss
        total_samples += accumulated_samples
    
    if total_samples == 0:
        return {'loss': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0, 'num_optimizer_steps': 0}
    
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    metrics['loss'] = total_loss / total_samples
    metrics['num_optimizer_steps'] = num_optimizer_steps
    metrics['skipped_batches'] = skipped_batches
    metrics['avg_batch_time_ms'] = np.mean(batch_times) * 1000 if batch_times else 0
    metrics['avg_data_time_ms'] = np.mean(data_times) * 1000 if data_times else 0
    metrics['avg_compute_time_ms'] = np.mean(compute_times) * 1000 if compute_times else 0
    metrics['throughput'] = total_samples / sum(batch_times) if batch_times else 0
    
    return metrics


@torch.no_grad()
def validate(
    model, 
    loader, 
    criterion, 
    device: torch.device, 
    preprocessor, 
    use_amp: bool = True
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(batch)
        
        edge_labels = batch.edge_labels
        preds_flat = preds.squeeze(-1) if preds.dim() > 1 else preds
        labels_flat = edge_labels.squeeze(-1) if edge_labels.dim() > 1 else edge_labels
        
        if preds_flat.numel() == 0:
            continue
        
        min_len = min(len(preds_flat), len(labels_flat))
        preds_flat, labels_flat = preds_flat[:min_len], labels_flat[:min_len]
        
        if torch.isnan(preds_flat).any() or torch.isinf(preds_flat).any():
            continue
        
        loss = criterion(preds_flat.float(), labels_flat.float())
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        bs = preds_flat.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        
        all_preds.append(preprocessor.inverse_transform_time(preds_flat.float().cpu().numpy()))
        all_targets.append(preprocessor.inverse_transform_time(labels_flat.float().cpu().numpy()))
    
    if total_samples == 0:
        return {'loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'mape': 0.0, 'r2': 0.0}
    
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    metrics['loss'] = total_loss / total_samples
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    config_path = os.environ.get('SM_HP_CONFIG_PATH', 's3://graph-transformer-exp/configs/config.json')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    log = get_logger(rank)
    
    log.info(f"Loading config from: {config_path}")
    config = Config.load(config_path)
    
    set_seed(config.training.seed, rank)
    shutdown = GracefulShutdown()
    
    log.info("=" * 70)
    log.info(f"CAUSAL GRAPH TRANSFORMER TRAINING: {config.experiment_name}")
    log.info("=" * 70)
    log.info(f"  World size: {world_size}, Rank: {rank}, Device: {device}")
    log.info(f"  Epochs: {config.training.epochs}, Batch size: {config.training.batch_size}")
    log.info(f"  LR: {config.training.learning_rate}, Hidden dim: {config.model.hidden_dim}")
    log.info(f"  Layers: {config.model.num_layers}, Heads: {config.model.num_heads}")
    log.info(f"  Shared Memory: ENABLED")
    log.info("=" * 70)
    
    # Initialize S3 uploader (rank 0 only)
    s3_uploader = None
    if rank == 0:
        s3_uploader = get_s3_uploader(config, log)
    
    train_ds, val_ds, test_ds = None, None, None
    preprocessor_local_path = None
    
    try:
        # Load data into shared memory
        train_ds, val_ds, test_ds, preprocessor = load_data_shared_memory(
            config, rank, local_rank, world_size, log
        )
        
        # Store preprocessor path for S3 upload
        preprocessor_local_path = os.path.join(model_dir, 'preprocessor.pkl')
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Create dataloaders
        train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
            train_ds, val_ds, test_ds, config, rank, local_rank, world_size
        )
        
        # Create model
        model, vocab_sizes, feature_dims = create_model(
            preprocessor, config, device, local_rank, world_size, rank
        )
        
        if rank == 0:
            m = model.module if hasattr(model, 'module') else model
            params = m.get_num_parameters()
            log.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        
        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            eps=1e-8,
            fused=True
        )
        
        criterion = nn.MSELoss()
        
        early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        warmup_epochs = config.training.warmup_epochs
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.epochs - warmup_epochs,
                eta_min=config.training.learning_rate * config.training.min_lr_ratio
            )
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.epochs,
                eta_min=config.training.learning_rate * config.training.min_lr_ratio
            )
        
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_epoch = 0
        
        log.info("=" * 70)
        log.info("TRAINING LOOP (Shared Memory Mode)")
        log.info("=" * 70)
        
        # Training Loop
        for epoch in range(1, config.training.epochs + 1):
            if shutdown.should_stop:
                log.info("Shutdown requested")
                break
            
            epoch_start = time.time()
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, 
                preprocessor, config, rank, world_size, epoch
            )
            sync_barrier(device)
            
            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device, preprocessor, config.training.use_amp
            )
            
            scheduler.step()
            
            # Reduce metrics across workers
            if world_size > 1:
                train_metrics = reduce_metrics(train_metrics, device, world_size)
                val_metrics = reduce_metrics(val_metrics, device, world_size)
            
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            log.info(f"\nEpoch {epoch}/{config.training.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
            log.info(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}h, R²: {train_metrics['r2']:.4f}")
            log.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}h, R²: {val_metrics['r2']:.4f}")
            log.info(f"  Throughput: {train_metrics.get('throughput', 0):.0f} samples/s | "
                    f"Data: {train_metrics.get('avg_data_time_ms', 0):.1f}ms | "
                    f"Compute: {train_metrics.get('avg_compute_time_ms', 0):.1f}ms")
            
            if train_metrics.get('skipped_batches', 0) > 0:
                log.warning(f"  Skipped {train_metrics['skipped_batches']} batches due to NaN/Inf")
            
            # Save best model (rank 0 only)
            if rank == 0:
                val_loss = val_metrics['loss']
                val_mae = val_metrics['mae']
                
                if val_loss < float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    best_epoch = epoch
                    
                    local_best_path = os.path.join(model_dir, 'best_model.pt')
                    
                    # Save locally and upload to S3
                    save_and_upload_best_model(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics={'val_loss': val_loss, 'val_mae': val_mae, 'val_r2': val_metrics['r2']},
                        vocab_sizes=vocab_sizes,
                        feature_dims=feature_dims,
                        config=config,
                        local_path=local_best_path,
                        s3_uploader=s3_uploader,
                        preprocessor_path=preprocessor_local_path,
                        log=log
                    )
                    
                    log.info(f"  ✓ New best model! (MAE: {best_val_mae:.2f}h)")
                    if s3_uploader:
                        log.info(f"  ✓ Uploaded to {s3_uploader.best_model_s3_path}")
            
            # Early stopping
            should_stop = torch.tensor([0], device=device)
            if rank == 0 and early_stopping(val_metrics['loss']):
                log.info(f"Early stopping triggered at epoch {epoch}")
                should_stop[0] = 1
            
            if world_size > 1:
                dist.broadcast(should_stop, src=0)
            if should_stop.item() == 1:
                break
        
        sync_barrier(device)
        
        # Final Test Evaluation
        if rank == 0:
            log.info("=" * 70)
            log.info("FINAL TEST EVALUATION")
            log.info("=" * 70)
            
            best_path = os.path.join(model_dir, 'best_model.pt')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                m = model.module if hasattr(model, 'module') else model
                m.load_state_dict(ckpt['model_state_dict'])
                
                test_metrics = validate(
                    model, test_loader, criterion, device, preprocessor, config.training.use_amp
                )
                
                log.info(f"Test Results:")
                log.info(f"  MAE:  {test_metrics['mae']:.2f} hours")
                log.info(f"  RMSE: {test_metrics['rmse']:.2f} hours")
                log.info(f"  MAPE: {test_metrics['mape']:.2f}%")
                log.info(f"  R²:   {test_metrics['r2']:.4f}")
                
                results = {
                    'experiment_name': config.experiment_name,
                    'best_epoch': best_epoch,
                    'best_val_loss': float(best_val_loss),
                    'best_val_mae': float(best_val_mae),
                    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'config': config.to_dict()
                }
                
                local_results_path = os.path.join(model_dir, 'results.json')
                with open(local_results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Upload final results to S3
                if s3_uploader:
                    s3_key = f"{s3_uploader.prefix}/{s3_uploader.job_name}/best_model/results.json"
                    if s3_uploader.upload_file(local_results_path, s3_key):
                        log.info(f"  ✓ Results uploaded to S3")
            
            log.info("=" * 70)
            log.info(f"TRAINING COMPLETE!")
            log.info(f"  Best epoch: {best_epoch}")
            log.info(f"  Best Val MAE: {best_val_mae:.2f} hours")
            if s3_uploader:
                log.info(f"  Model saved to: {s3_uploader.best_model_s3_path}")
            log.info("=" * 70)
    
    except Exception as e:
        logging.error(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        for ds in [train_ds, val_ds, test_ds]:
            if ds is not None:
                try:
                    ds.close()
                except:
                    pass
        cleanup_distributed()


if __name__ == '__main__':
    main()