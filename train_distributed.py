#!/usr/bin/env python3
"""
train_distributed.py - Distributed training with S3 H5 cache
"""

import os
import sys
import json
import argparse
import signal
import gc
import time
from datetime import timedelta
from contextlib import nullcontext
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


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
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=40)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=32)
    
    # Data
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    
    # Paths
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output'))
    parser.add_argument('--training', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--distance_file', type=str, default='location_distances_complete.csv')
    parser.add_argument('--data_file', type=str, default='source.json')
    
    # S3 Cache
    parser.add_argument('--cache_dir', type=str, default="s3://graph-transformer-exp/cache000/",
                        help='S3 path for H5 cache (e.g., s3://bucket/path/)')
    parser.add_argument('--load_from_cache', type=bool, default=False)
    parser.add_argument('--save_to_cache', type=bool, default=True)
    
    # Distributed
    parser.add_argument('--find_unused_parameters', action='store_true', default=False)
    parser.add_argument('--checkpoint_frequency', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # AMP (autocast only, no grad scaling)
    parser.add_argument('--disable_amp', action='store_true', default=False,
                        help='Disable automatic mixed precision')
    
    return parser.parse_args()


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_distributed():
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
    if dist.is_initialized():
        dist.barrier()
        dummy = torch.zeros(1, device=device)
        dist.all_reduce(dummy)
        del dummy
        torch.cuda.synchronize()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int, rank: int = 0):
    actual_seed = seed + rank
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    np.random.seed(actual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GracefulShutdown:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.warning(f"Received signal {signum}, shutting down...")
        self.should_stop = True


class EarlyStopping:
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


# =============================================================================
# S3 UTILITIES
# =============================================================================

def s3_exists(s3_path: str) -> bool:
    """Check if S3 object exists."""
    import boto3
    from botocore.exceptions import ClientError
    
    path = s3_path.replace('s3://', '')
    bucket = path.split('/')[0]
    key = '/'.join(path.split('/')[1:])
    
    try:
        boto3.client('s3').head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def s3_upload(local_path: str, s3_path: str, log_fn=None):
    """Upload file to S3."""
    import boto3
    
    path = s3_path.replace('s3://', '')
    bucket = path.split('/')[0]
    key = '/'.join(path.split('/')[1:])
    
    if log_fn:
        log_fn(f"  Uploading {local_path} -> s3://{bucket}/{key}")
    
    boto3.client('s3').upload_file(local_path, bucket, key)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(args, rank, local_rank, world_size, log):
    """Load data with S3 H5 cache."""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    from config import Config
    
    device = torch.device(f'cuda:{local_rank}')
    
    # S3 paths
    cache_dir = args.cache_dir.rstrip('/')
    train_cache = f"{cache_dir}/train.h5"
    val_cache = f"{cache_dir}/val.h5"
    test_cache = f"{cache_dir}/test.h5"
    preprocessor_cache = f"{cache_dir}/preprocessor.pkl"
    
    if rank == 0:
        log.info("=" * 60)
        log.info("PREPARING DATA")
        log.info("=" * 60)
        log.info(f"  S3 cache dir: {cache_dir}")
    
    # Check if cache exists
    cache_exists = False
    if rank == 0:
        cache_exists = all([
            s3_exists(train_cache),
            s3_exists(val_cache),
            s3_exists(test_cache),
            s3_exists(preprocessor_cache)
        ])
        log.info(f"  Cache exists: {cache_exists}")
    
    # Broadcast cache_exists
    cache_tensor = torch.tensor([1 if cache_exists else 0], device=device)
    if world_size > 1:
        dist.broadcast(cache_tensor, src=0)
    cache_exists = cache_tensor.item() == 1
    
    # Rank 0 creates cache if needed
    #if rank == 0 and not (args.load_from_cache and cache_exists):
    #    log.info("Creating H5 cache...")
    #    os.makedirs(args.model_dir, exist_ok=True)
        
        # Load raw data
    #    data_path = os.path.join(args.training, args.data_file)
    #    log.info(f"  Loading: {data_path}")
    #    df = pd.read_json(data_path)
    #    log.info(f"  Samples: {len(df)}")
        
        # Distance matrix
    #    distance_path = os.path.join(args.training, args.distance_file)
    #    distance_df = pd.read_csv(distance_path) if os.path.exists(distance_path) else None
        
        # Split
    #    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    #    train_size = int(args.train_ratio * len(df))
    #    val_size = int(args.val_ratio * len(df))
        
    #    train_df = df.iloc[:train_size].reset_index(drop=True)
    #    val_df = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    #    test_df = df.iloc[train_size + val_size:].reset_index(drop=True)
        
    #    log.info(f"  Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    #    del df
    #    gc.collect()
        
        # Preprocessor
    #    config = Config()
    #    preprocessor = PackageLifecyclePreprocessor(config=config, distance_df=distance_df)
    #    preprocessor.fit(train_df)
        
    #    local_preprocessor = os.path.join(args.model_dir, 'preprocessor.pkl')
    #    preprocessor.save(local_preprocessor)
    #    s3_upload(local_preprocessor, preprocessor_cache, log.info)
        
    #    del distance_df
    #    gc.collect()
        
        # Create datasets (saves to S3)
    #    log.info("  Creating train.h5...")
    #    PackageLifecycleDataset(
    #        df=train_df, preprocessor=preprocessor,
    #        h5_cache_path=train_cache,
    #        load_from_cache=False, save_to_cache=True,
    #        log_fn=log.info
    #    )
    #    del train_df
    #    gc.collect()
        
    #    log.info("  Creating val.h5...")
    #    PackageLifecycleDataset(
    #        df=val_df, preprocessor=preprocessor,
    #        h5_cache_path=val_cache,
    #        load_from_cache=False, save_to_cache=True,
    #        log_fn=log.info
    #    )
    #    del val_df
    #    gc.collect()
        
    #    log.info("  Creating test.h5...")
    #    PackageLifecycleDataset(
    #        df=test_df, preprocessor=preprocessor,
    #        h5_cache_path=test_cache,
    #        load_from_cache=False, save_to_cache=True,
    #        log_fn=log.info
    #    )
    #    del test_df
    #    gc.collect()
        
    #    log.info("  ✓ Cache created")
    
    # Wait for rank 0
    sync_barrier(device)
    
    if rank == 0:
        log.info("Loading datasets from S3 cache...")
    
    # All ranks load from S3 (dataset handles download)
    log_fn = log.info if rank == 0 else None
    
    train_dataset = PackageLifecycleDataset(
        h5_cache_path=train_cache,
        load_from_cache=True,
        save_to_cache =False,
        log_fn=log_fn
    )
    
    val_dataset = PackageLifecycleDataset(
        h5_cache_path=val_cache,
        load_from_cache=True,
        save_to_cache =False,
        log_fn=log_fn
    )
    
    test_dataset = PackageLifecycleDataset(
        h5_cache_path=test_cache,
        load_from_cache=True,
        save_to_cache =False,
        log_fn=log_fn
    )
    
    # Load preprocessor
    local_preprocessor = os.path.join(args.model_dir, 'preprocessor.pkl')
    if not os.path.exists(local_preprocessor):
        import boto3
        path = preprocessor_cache.replace('s3://', '')
        bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])
        os.makedirs(os.path.dirname(local_preprocessor), exist_ok=True)
        boto3.client('s3').download_file(bucket, key, local_preprocessor)
    
    preprocessor = PackageLifecyclePreprocessor.load(local_preprocessor)
    
    sync_barrier(device)
    
    if rank == 0:
        log.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        log.info("=" * 60)
    
    return train_dataset, val_dataset, test_dataset, preprocessor


def create_dataloaders(train_dataset, val_dataset, test_dataset, args, rank, world_size):
    log = get_logger(rank)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), drop_last=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, drop_last=False,
        collate_fn=val_dataset.get_collate_fn(),
        num_workers=args.num_workers, pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        shuffle=False, drop_last=False,
        collate_fn=test_dataset.get_collate_fn(),
        num_workers=args.num_workers, pin_memory=True,
    )
    
    log.info(f"DataLoaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader, train_sampler


# =============================================================================
# MODEL
# =============================================================================

def create_model(preprocessor, args, device, local_rank, world_size, rank):
    from config import ModelConfig
    from models.event_predictor import EventTimePredictor
    
    log = get_logger(rank)
    vocab_sizes = preprocessor.get_vocab_sizes()
    
    model_config = ModelConfig.from_preprocessor(
        preprocessor,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
        output_dim=1,
        use_edge_features=True,
        use_global_attention=False,
        use_positional_encoding=False
    )
    
    model = EventTimePredictor(model_config, vocab_sizes).to(device)
    
    # Initialize weights for stability
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    model.apply(init_weights)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters,
                    gradient_as_bucket_view=True)
        log.info("Model wrapped with DDP")
    
    return model, model_config, vocab_sizes


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, targets):
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


def reduce_metrics(metrics, device, world_size):
    if world_size <= 1:
        return metrics
    reduced = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            t = torch.tensor([1.0 if v else 0.0], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = t.item() > 0
        elif isinstance(v, (int, float)):
            t = torch.tensor(float(v), device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = t.item() / world_size
        else:
            reduced[k] = v
    return reduced


# =============================================================================
# TRAINING
# =============================================================================

def diagnose_batch(model, batch, device, rank, use_amp=True):
    """Run diagnostic on a single batch to check for numerical issues."""
    if rank != 0:
        return
    
    model.eval()
    with torch.no_grad():
        batch = batch.to(device, non_blocking=True)
        
        # Check input data
        logging.info("  [Diagnostic] Input checks:")
        logging.info(f"    - Num nodes: {batch.num_nodes}")
        logging.info(f"    - Num edges: {batch.edge_index.shape[1] if batch.edge_index is not None else 0}")
        
        # Check for NaN/Inf in input features
        if hasattr(batch, 'x') and batch.x is not None:
            x_nan = torch.isnan(batch.x).any().item()
            x_inf = torch.isinf(batch.x).any().item()
            logging.info(f"    - Node features: shape={batch.x.shape}, has_nan={x_nan}, has_inf={x_inf}")
            if x_nan or x_inf:
                logging.warning("    ⚠ NaN/Inf detected in input features!")
        
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            e_nan = torch.isnan(batch.edge_attr).any().item()
            e_inf = torch.isinf(batch.edge_attr).any().item()
            logging.info(f"    - Edge features: shape={batch.edge_attr.shape}, has_nan={e_nan}, has_inf={e_inf}")
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            try:
                preds = model(batch)
                p_nan = torch.isnan(preds).any().item()
                p_inf = torch.isinf(preds).any().item()
                logging.info(f"  [Diagnostic] Predictions: min={preds.min():.4f}, max={preds.max():.4f}, "
                           f"mean={preds.mean():.4f}, has_nan={p_nan}, has_inf={p_inf}")
                
                if p_nan or p_inf:
                    logging.warning("  ⚠ NaN/Inf detected in predictions!")
                    
            except Exception as e:
                logging.error(f"  [Diagnostic] Forward pass failed: {e}")
    
    model.train()


def train_epoch(model, loader, optimizer, criterion, device, preprocessor,
                grad_accum=1, rank=0, world_size=1, max_grad_norm=1.0,
                epoch=1, use_amp=True):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    num_optimizer_steps = 0
    valid_batches = 0
    skipped_batches = 0
    skipped_steps_due_to_nan = 0
    skipped_due_to_invalid_loss = 0
    
    is_ddp = world_size > 1 and hasattr(model, 'no_sync')
    num_batches = len(loader)
    
    log = get_logger(rank)
    
    if num_batches == 0:
        return {
            'loss': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0,
            'num_optimizer_steps': 0, 'valid_batches': 0, 'skipped_batches': 0,
            'skipped_steps_due_to_nan': 0, 'skipped_due_to_invalid_loss': 0
        }
    
    # Run diagnostic on first batch of first epoch
    if epoch == 1 and rank == 0:
        for batch in loader:
            diagnose_batch(model, batch, device, rank, use_amp)
            break
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_samples = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        should_sync = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches
        ctx = model.no_sync() if (is_ddp and not should_sync) else nullcontext()
        
        with ctx:
            # Use autocast for forward pass (memory savings)
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(batch)
                mask = batch.label_mask
                masked_preds = preds[mask].squeeze(-1) if preds[mask].dim() > 1 else preds[mask]
                masked_targets = batch.labels.squeeze(-1) if batch.labels.dim() > 1 else batch.labels
                
                if masked_preds.numel() == 0:
                    skipped_batches += 1
                    continue
                
                min_len = min(len(masked_preds), len(masked_targets))
                masked_preds, masked_targets = masked_preds[:min_len], masked_targets[:min_len]
                
                # Check for NaN/Inf in predictions
                if torch.isnan(masked_preds).any() or torch.isinf(masked_preds).any():
                    skipped_due_to_invalid_loss += 1
                    if rank == 0 and skipped_due_to_invalid_loss <= 5:
                        log.warning(f"  Batch {batch_idx}: NaN/Inf in predictions, skipping")
                    continue
                
                # Check for NaN/Inf in targets
                if torch.isnan(masked_targets).any() or torch.isinf(masked_targets).any():
                    skipped_due_to_invalid_loss += 1
                    if rank == 0 and skipped_due_to_invalid_loss <= 5:
                        log.warning(f"  Batch {batch_idx}: NaN/Inf in targets, skipping")
                    continue
                
                # Compute loss in float32 for stability
                loss = criterion(masked_preds.float(), masked_targets.float())
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    skipped_due_to_invalid_loss += 1
                    if rank == 0 and skipped_due_to_invalid_loss <= 5:
                        log.warning(f"  Batch {batch_idx}: Loss is {loss.item()}, skipping")
                    continue
                
                scaled_loss = loss / grad_accum
            
            # Backward pass (outside autocast for full precision gradients)
            scaled_loss.backward()
        
        valid_batches += 1
        bs = masked_preds.size(0)
        accumulated_loss += loss.item() * bs
        accumulated_samples += bs
        
        # Store predictions for metrics
        all_preds.append(preprocessor.inverse_transform_time(masked_preds.detach().float().cpu().numpy()))
        all_targets.append(preprocessor.inverse_transform_time(masked_targets.float().cpu().numpy()))
        
        if should_sync:
            # Compute gradient norm and clip
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Check for NaN/Inf gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                skipped_steps_due_to_nan += 1
                if rank == 0 and skipped_steps_due_to_nan <= 5:
                    log.warning(f"  Step {batch_idx}: Gradient norm is {grad_norm}, skipping optimizer step")
                optimizer.zero_grad()
                continue
            
            # Perform optimizer step
            optimizer.step()
            num_optimizer_steps += 1
            optimizer.zero_grad()
            
            # Accumulate totals
            total_loss += accumulated_loss
            total_samples += accumulated_samples
            accumulated_loss = 0.0
            accumulated_samples = 0
    
    # Handle any remaining accumulated samples
    if accumulated_samples > 0:
        total_loss += accumulated_loss
        total_samples += accumulated_samples
    
    if total_samples == 0:
        return {
            'loss': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0,
            'num_optimizer_steps': num_optimizer_steps,
            'valid_batches': valid_batches,
            'skipped_batches': skipped_batches,
            'skipped_steps_due_to_nan': skipped_steps_due_to_nan,
            'skipped_due_to_invalid_loss': skipped_due_to_invalid_loss
        }
    
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    metrics['loss'] = total_loss / total_samples
    metrics['num_optimizer_steps'] = num_optimizer_steps
    metrics['valid_batches'] = valid_batches
    metrics['skipped_batches'] = skipped_batches
    metrics['skipped_steps_due_to_nan'] = skipped_steps_due_to_nan
    metrics['skipped_due_to_invalid_loss'] = skipped_due_to_invalid_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor, use_amp=True):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(batch)
        
        mask = batch.label_mask
        
        masked_preds = preds[mask].squeeze(-1) if preds[mask].dim() > 1 else preds[mask]
        masked_targets = batch.labels.squeeze(-1) if batch.labels.dim() > 1 else batch.labels
        
        if masked_preds.numel() == 0:
            continue
        
        min_len = min(len(masked_preds), len(masked_targets))
        masked_preds, masked_targets = masked_preds[:min_len], masked_targets[:min_len]
        
        # Skip batches with NaN/Inf
        if torch.isnan(masked_preds).any() or torch.isinf(masked_preds).any():
            continue
        
        # Compute loss in float32
        loss = criterion(masked_preds.float(), masked_targets.float())
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        bs = masked_preds.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        
        all_preds.append(preprocessor.inverse_transform_time(masked_preds.float().cpu().numpy()))
        all_targets.append(preprocessor.inverse_transform_time(masked_targets.float().cpu().numpy()))
    
    if total_samples == 0:
        return {'loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'mape': 0.0, 'r2': 0.0}
    
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    metrics['loss'] = total_loss / total_samples
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics,
                    model_config, vocab_sizes, save_path, is_best=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config.to_dict(),
        'vocab_sizes': vocab_sizes,
        'metrics': metrics,
        'is_best': is_best
    }
    temp_path = save_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, save_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    shutdown = GracefulShutdown()
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(args.seed, rank)
    
    log = get_logger(rank)
    
    use_amp = not args.disable_amp
    
    log.info("=" * 60)
    log.info("DISTRIBUTED TRAINING")
    log.info("=" * 60)
    log.info(f"  World size: {world_size}, Rank: {rank}, Device: {device}")
    log.info(f"  S3 cache: {args.cache_dir}")
    log.info(f"  AMP (autocast): {use_amp}")
    log.info(f"  Warmup epochs: {args.warmup_epochs}")
    log.info("=" * 60)
    
    try:
        train_ds, val_ds, test_ds, preprocessor = load_and_prepare_data(
            args, rank, local_rank, world_size, log
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
            train_ds, val_ds, test_ds, args, rank, world_size
        )
        
        model, model_config, vocab_sizes = create_model(
            preprocessor, args, device, local_rank, world_size, rank
        )
        
        if rank == 0:
            m = model.module if hasattr(model, 'module') else model
            params = m.get_num_parameters()
            log.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        
        # Learning rate scheduler with warmup
        warmup_epochs = args.warmup_epochs
        if warmup_epochs > 0:
            # Warmup scheduler: start from 1% of LR and increase to 100%
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            # Main scheduler after warmup
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs,
                eta_min=args.learning_rate * 0.01
            )
            # Combined scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            log.info(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> CosineAnnealingLR")
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.learning_rate * 0.01
            )
            log.info("Scheduler: CosineAnnealingLR (no warmup)")
        
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_epoch = 0
        total_optimizer_steps = 0
        
        log.info("=" * 60)
        log.info("TRAINING")
        log.info("=" * 60)
        
        for epoch in range(1, args.epochs + 1):
            if shutdown.should_stop:
                log.info("Shutdown requested")
                break
            
            epoch_start = time.time()
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Training
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, preprocessor,
                args.gradient_accumulation_steps, rank, world_size,
                max_grad_norm=args.max_grad_norm, epoch=epoch, use_amp=use_amp
            )
            sync_barrier(device)
            
            # Validation
            val_metrics = validate(model, val_loader, criterion, device, preprocessor, use_amp=use_amp)
            
            # Track optimizer steps
            epoch_optimizer_steps = train_metrics.get('num_optimizer_steps', 0)
            total_optimizer_steps += epoch_optimizer_steps
            
            # Step scheduler AFTER training
            scheduler.step()
            
            # Reduce metrics across workers
            if world_size > 1:
                train_metrics_to_reduce = {k: v for k, v in train_metrics.items() 
                                          if k not in ['num_optimizer_steps', 'valid_batches', 
                                                       'skipped_batches', 'skipped_steps_due_to_nan',
                                                       'skipped_due_to_invalid_loss']}
                val_metrics_to_reduce = {k: v for k, v in val_metrics.items()}
                
                train_metrics_reduced = reduce_metrics(train_metrics_to_reduce, device, world_size)
                val_metrics = reduce_metrics(val_metrics_to_reduce, device, world_size)
                
                # Keep original counts (sum them across workers)
                for key in ['num_optimizer_steps', 'valid_batches', 'skipped_batches', 
                           'skipped_steps_due_to_nan', 'skipped_due_to_invalid_loss']:
                    if key in train_metrics:
                        t = torch.tensor(float(train_metrics[key]), device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        train_metrics_reduced[key] = int(t.item())
                
                train_metrics = train_metrics_reduced
            
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            
            # Log epoch results
            log.info(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
            log.info(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}h, R²: {train_metrics['r2']:.4f}")
            log.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}h, R²: {val_metrics['r2']:.4f}")
            log.info(f"  Steps - Optimizer: {epoch_optimizer_steps}, Total: {total_optimizer_steps}")
            
            # Warnings for potential issues
            if train_metrics.get('skipped_batches', 0) > 0:
                log.warning(f"  ⚠ Skipped {train_metrics['skipped_batches']} batches (empty predictions)")
            
            if train_metrics.get('skipped_steps_due_to_nan', 0) > 0:
                log.warning(f"  ⚠ Skipped {train_metrics['skipped_steps_due_to_nan']} optimizer steps (inf/nan gradients)")
            
            if train_metrics.get('skipped_due_to_invalid_loss', 0) > 0:
                log.warning(f"  ⚠ Skipped {train_metrics['skipped_due_to_invalid_loss']} batches (invalid loss/predictions)")
            
            if epoch_optimizer_steps == 0:
                log.warning(f"  ⚠ No optimizer steps this epoch! Check data/model.")
            
            # Save best model
            if rank == 0:
                val_loss = val_metrics['loss']
                val_mae = val_metrics['mae']
                
                if val_loss < float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    best_epoch = epoch
                    
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        {'val_loss': val_loss, 'val_mae': val_mae, 'val_r2': val_metrics['r2']},
                        model_config, vocab_sizes,
                        os.path.join(args.model_dir, 'best_model.pt'), is_best=True
                    )
                    log.info(f"  ✓ New best model! (MAE: {best_val_mae:.2f}h)")
                
                # Periodic checkpoint
                if epoch % args.checkpoint_frequency == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        {'val_loss': val_loss, 'val_mae': val_mae},
                        model_config, vocab_sizes,
                        os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pt')
                    )
            
            # Early stopping
            should_stop = torch.tensor([0], device=device)
            if rank == 0:
                val_loss = val_metrics['loss']
                if val_loss < float('inf') and early_stopping(val_loss):
                    log.info(f"Early stopping at epoch {epoch}")
                    should_stop[0] = 1
            
            if world_size > 1:
                dist.broadcast(should_stop, src=0)
            if should_stop.item() == 1:
                break
        
        sync_barrier(device)
        
        # Final test evaluation
        if rank == 0:
            log.info("=" * 60)
            log.info("TESTING")
            log.info("=" * 60)
            
            best_path = os.path.join(args.model_dir, 'best_model.pt')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                m = model.module if hasattr(model, 'module') else model
                m.load_state_dict(ckpt['model_state_dict'])
                
                test_metrics = validate(model, test_loader, criterion, device, preprocessor, use_amp=use_amp)
                
                log.info(f"Test: MAE={test_metrics['mae']:.2f}h, RMSE={test_metrics['rmse']:.2f}h, R²={test_metrics['r2']:.4f}")
                
                # Save results
                results = {
                    'best_epoch': best_epoch,
                    'best_val_loss': float(best_val_loss),
                    'best_val_mae': float(best_val_mae),
                    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'total_optimizer_steps': total_optimizer_steps,
                    'args': vars(args)
                }
                
                with open(os.path.join(args.model_dir, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                log.warning("No best model found!")
            
            log.info("=" * 60)
            log.info(f"COMPLETE! Best epoch: {best_epoch}, Val MAE: {best_val_mae:.2f}h")
            log.info("=" * 60)
    
    except Exception as e:
        logging.error(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()