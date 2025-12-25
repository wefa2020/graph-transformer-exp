#!/usr/bin/env python3
"""
sagemaker_train.py - Distributed training script for SageMaker
"""

import os
import sys
import json
import argparse
import signal
from datetime import timedelta
from contextlib import nullcontext
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def configure_distributed_environment():
    """Configure environment variables for EFA/NCCL"""
    env_vars = {
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "FI_PROVIDER": "efa",
        "FI_EFA_FORK_SAFE": "1",
        "RDMAV_FORK_SAFE": "1",
        "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_TIMEOUT": "1800",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


def get_num_workers():
    """Calculate optimal number of workers for DataLoader"""
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_distributed():
    """Initialize distributed training"""
    configure_distributed_environment()
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30)
        )
        torch.cuda.set_device(local_rank)
        
        # Verify communication
        test_tensor = torch.ones(1).cuda(local_rank)
        dist.all_reduce(test_tensor)
        dist.barrier()
        
        if rank == 0:
            logger.info(f"Distributed training initialized: {world_size} GPUs")
    else:
        logger.info("Single GPU mode")
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training - simple version without barrier"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

class GracefulShutdown:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int, rank: int = 0):
    """Set random seed for reproducibility"""
    actual_seed = seed + rank
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    np.random.seed(actual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=32)
    
    # Data parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    
    # SageMaker paths
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output'))
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--distance_file', type=str, default='location_distances_complete.csv')
    parser.add_argument('--data_file', type=str, default='test.json')
    
    # Distributed training options
    parser.add_argument('--find_unused_parameters', action='store_true', default=True)
    parser.add_argument('--checkpoint_frequency', type=int, default=5)
    
    return parser.parse_args()


def load_data(args, rank):
    """Load data from JSON file"""
    if rank == 0:
        logger.info(f"Loading data from: {args.training}")
    
    data_path = os.path.join(args.training, args.data_file)
    df = pd.read_json(data_path)
    
    if rank == 0:
        logger.info(f"Loaded {len(df)} package lifecycles")
    
    return df


def split_data(df, args, rank):
    """Split data into train/val/test sets"""
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    train_size = int(args.train_ratio * len(df))
    val_size = int(args.val_ratio * len(df))
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    if rank == 0:
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_preprocessor(args, train_df, rank):
    """Create and fit the preprocessor"""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from config import Config
    
    distance_file_path = os.path.join(args.training, args.distance_file)
    if not os.path.exists(distance_file_path):
        distance_file_path = None
        if rank == 0:
            logger.warning("Distance file not found, proceeding without it")
    
    config = Config()
    preprocessor = PackageLifecyclePreprocessor(
        config=config,
        distance_file_path=distance_file_path
    )
    preprocessor.fit(train_df)
    
    if rank == 0:
        logger.info(f"Feature dimensions: {preprocessor.get_feature_dimensions()}")
    
    return preprocessor


def create_dataloaders(train_df, val_df, test_df, preprocessor, args, rank, world_size):
    """Create distributed data loaders"""
    from data.dataset import PackageLifecycleDataset
    
    train_dataset = PackageLifecycleDataset(train_df, preprocessor, return_labels=True)
    val_dataset = PackageLifecycleDataset(val_df, preprocessor, return_labels=True)
    test_dataset = PackageLifecycleDataset(test_df, preprocessor, return_labels=True)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    num_workers = get_num_workers()
    
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        shuffle=False, **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader, train_sampler


def create_model(preprocessor, args, device, local_rank, world_size):
    """Create model with distributed wrapper"""
    from config import ModelConfig
    from models.event_predictor import EventTimePredictor
    
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
        use_global_attention=True,
        use_positional_encoding=True
    )
    
    model = EventTimePredictor(model_config, vocab_sizes)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
            gradient_as_bucket_view=True,
        )
    
    return model, model_config, vocab_sizes


def compute_metrics(preds, targets):
    """Compute evaluation metrics"""
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100 if mask.sum() > 0 else 0.0
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def reduce_metrics(metrics, device, world_size):
    """Reduce metrics across all ranks"""
    if world_size <= 1:
        return metrics
    
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item() / world_size
    
    return reduced


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, preprocessor, 
                scaler=None, gradient_accumulation_steps=1, rank=0, world_size=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds, all_targets = [], []
    
    use_amp = scaler is not None
    is_ddp = world_size > 1 and hasattr(model, 'no_sync')
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        should_sync = (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
        sync_context = model.no_sync() if (is_ddp and not should_sync) else nullcontext()
        
        with sync_context:
            with torch.amp.autocast('cuda', enabled=use_amp):
                predictions = model(batch)
                mask = batch.label_mask
                masked_preds = predictions[mask]
                masked_targets = batch.labels
                loss = criterion(masked_preds, masked_targets) / gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        if should_sync:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        batch_size = masked_preds.size(0)
        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        total_samples += batch_size
        
        preds_hours = preprocessor.inverse_transform_time(masked_preds.detach().cpu().numpy())
        targets_hours = preprocessor.inverse_transform_time(masked_targets.detach().cpu().numpy())
        all_preds.append(preds_hours)
        all_targets.append(targets_hours)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / total_samples if total_samples > 0 else 0
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds, all_targets = [], []
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        predictions = model(batch)
        mask = batch.label_mask
        masked_preds = predictions[mask]
        masked_targets = batch.labels
        loss = criterion(masked_preds, masked_targets)
        
        batch_size = masked_preds.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        preds_hours = preprocessor.inverse_transform_time(masked_preds.cpu().numpy())
        targets_hours = preprocessor.inverse_transform_time(masked_targets.cpu().numpy())
        all_preds.append(preds_hours)
        all_targets.append(targets_hours)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / total_samples if total_samples > 0 else 0
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, 
                   model_config, vocab_sizes, save_path, is_best=False):
    """Save model checkpoint"""
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'model_config': model_config.to_dict(),
        'vocab_sizes': vocab_sizes,
        'metrics': metrics,
        'is_best': is_best
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function"""
    args = parse_args()
    shutdown_handler = GracefulShutdown()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(args.seed, rank)
    
    if rank == 0:
        logger.info(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        logger.info(f"Args: {args}")
    
    # Load data
    df = load_data(args, rank)
    train_df, val_df, test_df = split_data(df, args, rank)
    
    # Create preprocessor
    preprocessor = create_preprocessor(args, train_df, rank)
    
    if world_size > 1:
        dist.barrier()
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
        train_df, val_df, test_df, preprocessor, args, rank, world_size
    )
    
    # Create model
    model, model_config, vocab_sizes = create_model(preprocessor, args, device, local_rank, world_size)
    
    if rank == 0:
        model_for_params = model.module if hasattr(model, 'module') else model
        params = model_for_params.get_num_parameters()
        logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Optimizer, scheduler, criterion
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    criterion = nn.MSELoss()
    
    # Early stopping
    from utils.metrics import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')
    if rank == 0:
        logger.info("Mixed precision training enabled")
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        if shutdown_handler.should_stop:
            logger.info(f"Rank {rank}: Graceful shutdown requested")
            break
        
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, preprocessor,
            scaler, args.gradient_accumulation_steps, rank, world_size
        )
        
        if world_size > 1:
            dist.barrier()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, preprocessor)
        scheduler.step()
        
        # Reduce metrics
        if world_size > 1:
            train_metrics = reduce_metrics(train_metrics, device, world_size)
            val_metrics = reduce_metrics(val_metrics, device, world_size)
        
        if rank == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}h | "
                       f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}h, R²: {val_metrics['r2']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_val_mae = val_metrics['mae']
                best_epoch = epoch
                
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    {'val_loss': val_metrics['loss'], 'val_mae': val_metrics['mae'], 'val_r2': val_metrics['r2']},
                    model_config, vocab_sizes,
                    os.path.join(args.model_dir, 'best_model.pt'),
                    is_best=True
                )
                preprocessor.save(os.path.join(args.model_dir, 'preprocessor.pkl'))
            
            # Periodic checkpoint
            if epoch % args.checkpoint_frequency == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    {'val_loss': val_metrics['loss'], 'val_mae': val_metrics['mae']},
                    model_config, vocab_sizes,
                    os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        
        # Early stopping
        should_stop = torch.tensor([0], device=device)
        if rank == 0 and early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            should_stop[0] = 1
        
        if world_size > 1:
            dist.broadcast(should_stop, src=0)
        
        if should_stop.item() == 1:
            break
    
    # Final sync before testing
    if world_size > 1:
        torch.cuda.synchronize()
        dist.barrier()
    
    # Test best model (rank 0 only)
    if rank == 0:
        logger.info("Testing best model...")
        checkpoint = torch.load(os.path.join(args.model_dir, 'best_model.pt'), map_location=device, weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = validate(model, test_loader, criterion, device, preprocessor)
        
        logger.info(f"Test Results - Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f}h, "
                   f"RMSE: {test_metrics['rmse']:.2f}h, MAPE: {test_metrics['mape']:.2f}%, R²: {test_metrics['r2']:.4f}")
        
        # Save results
        results = {
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'best_val_mae': float(best_val_mae),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'total_epochs': epoch,
            'args': vars(args)
        }
        with open(os.path.join(args.model_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(args.model_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config.to_dict(), f, indent=2)
        
        logger.info(f"Training complete! Best epoch: {best_epoch}, Val MAE: {best_val_mae:.2f}h, Test MAE: {test_metrics['mae']:.2f}h")
    
    # Clean up - simple destroy without barrier to avoid timeout
    cleanup_distributed()


if __name__ == '__main__':
    main()