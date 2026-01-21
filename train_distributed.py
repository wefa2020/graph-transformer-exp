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
# CONSTANTS
# =============================================================================

S3_OUTPUT_BUCKET = "graph-transformer-exp"
S3_OUTPUT_PREFIX = "output"


# =============================================================================
# MULTIPROCESSING SETUP (MUST BE FIRST)
# =============================================================================

def setup_multiprocessing():
    """Set multiprocessing start method to 'fork' for shared memory inheritance."""
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


def get_logger(rank: int) -> RankLogger:
    return RankLogger(rank)


# =============================================================================
# S3 UPLOAD
# =============================================================================

def upload_to_s3(local_path: str, bucket: str, s3_key: str, log: RankLogger) -> bool:
    """Upload a file to S3."""
    try:
        import boto3
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket, s3_key)
        log.info(f"  ✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        log.error(f"Failed to upload to S3: {e}")
        return False


def upload_best_model(
    local_model_path: str,
    job_name: str,
    log: RankLogger,
    preprocessor_path: Optional[str] = None
) -> bool:
    """Upload best model and related files to S3."""
    if not os.path.exists(local_model_path):
        log.error(f"Model file not found: {local_model_path}")
        return False
    
    s3_base = f"{S3_OUTPUT_PREFIX}/{job_name}/best_model"
    log.info(f"Uploading to s3://{S3_OUTPUT_BUCKET}/{s3_base}/")
    
    # Upload model
    success = upload_to_s3(
        local_model_path, 
        S3_OUTPUT_BUCKET, 
        f"{s3_base}/best_model.pt", 
        log
    )
    
    # Upload preprocessor if available
    if preprocessor_path and os.path.exists(preprocessor_path):
        upload_to_s3(
            preprocessor_path, 
            S3_OUTPUT_BUCKET, 
            f"{s3_base}/preprocessor.pkl", 
            log
        )
    
    return success


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
    local_path: str
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
    }
    
    temp_path = local_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, local_path)


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
    data_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
        data_time = time.time() - data_start
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
        batch_times.append(data_time + compute_time)
        
        # Log progress
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            avg_time = np.mean(batch_times[-100:])
            throughput = bs / avg_time if avg_time > 0 else 0
            logging.info(f"  Batch {batch_idx + 1}/{num_batches} | Throughput: {throughput:.0f} samples/s")
        
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
    job_name = os.environ.get('SM_TRAINING_JOB_NAME', f"training_{int(time.time())}")
    
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
    log.info(f"  Job name: {job_name}")
    log.info(f"  S3 output: s3://{S3_OUTPUT_BUCKET}/{S3_OUTPUT_PREFIX}/{job_name}/best_model/")
    log.info(f"  Epochs: {config.training.epochs}, Batch size: {config.training.batch_size}")
    log.info(f"  LR: {config.training.learning_rate}, Hidden dim: {config.model.hidden_dim}")
    log.info("=" * 70)
    
    train_ds, val_ds, test_ds = None, None, None
    preprocessor_local_path = os.path.join(model_dir, 'preprocessor.pkl')
    
    try:
        # Load data
        train_ds, val_ds, test_ds, preprocessor = load_data_shared_memory(
            config, rank, local_rank, world_size, log
        )
        
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
        local_best_path = os.path.join(model_dir, 'best_model.pt')
        
        log.info("=" * 70)
        log.info("TRAINING LOOP")
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
            
            # Reduce metrics
            if world_size > 1:
                train_metrics = reduce_metrics(train_metrics, device, world_size)
                val_metrics = reduce_metrics(val_metrics, device, world_size)
            
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            log.info(f"\nEpoch {epoch}/{config.training.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
            log.info(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}h, R²: {train_metrics['r2']:.4f}")
            log.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}h, R²: {val_metrics['r2']:.4f}")
            
            # Save best model (rank 0 only)
            if rank == 0:
                val_loss = val_metrics['loss']
                val_mae = val_metrics['mae']
                
                if val_loss < float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    best_epoch = epoch
                    
                    # Save locally
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        {'val_loss': val_loss, 'val_mae': val_mae, 'val_r2': val_metrics['r2']},
                        vocab_sizes, feature_dims, config, local_best_path
                    )
                    log.info(f"  ✓ New best model! (MAE: {best_val_mae:.2f}h)")
                    
                    # Upload to S3
                    upload_best_model(local_best_path, job_name, log, preprocessor_local_path)
            
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
            
            if os.path.exists(local_best_path):
                ckpt = torch.load(local_best_path, map_location=device, weights_only=False)
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
                
                # Save results
                results = {
                    'experiment_name': config.experiment_name,
                    'best_epoch': best_epoch,
                    'best_val_loss': float(best_val_loss),
                    'best_val_mae': float(best_val_mae),
                    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                }
                
                results_path = os.path.join(model_dir, 'results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Upload results to S3
                upload_to_s3(
                    results_path, 
                    S3_OUTPUT_BUCKET, 
                    f"{S3_OUTPUT_PREFIX}/{job_name}/best_model/results.json", 
                    log
                )
            
            log.info("=" * 70)
            log.info(f"TRAINING COMPLETE!")
            log.info(f"  Best epoch: {best_epoch}")
            log.info(f"  Best Val MAE: {best_val_mae:.2f} hours")
            log.info(f"  Model: s3://{S3_OUTPUT_BUCKET}/{S3_OUTPUT_PREFIX}/{job_name}/best_model/")
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