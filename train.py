# train.py - Training script with fixed progress bars and AMP

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm.auto import tqdm
import os
import sys
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from config import Config
from data.neptune_extractor import NeptuneDataExtractor
from data.data_preprocessor import PackageLifecyclePreprocessor
from data.dataset import PackageLifecycleDataset
from config import ModelConfig
from models.event_predictor import EventTimePredictor
from utils.metrics import compute_metrics,  EarlyStopping
from utils.package_filter import PackageEventValidator



def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(config) -> pd.DataFrame:
    """Load data from source"""
    print("\n" + "="*80)
    print("STEP 1: Load data")
    print("="*80)
    
    # Load directly from JSON
    data_path = getattr(config.data, 'source_file', 'data/graph-data/package_lifecycles_batch_2.json')
    df = pd.read_json(data_path)
    
    print(f"Loaded {len(df)} package lifecycles from {data_path}")
    
    # Print sample statistics
    if 'events' in df.columns:
        event_counts = df['events'].apply(len)
        print(f"Events per package: min={event_counts.min()}, max={event_counts.max()}, mean={event_counts.mean():.1f}")
    
    return df


def split_data(df: pd.DataFrame, config) -> tuple:
    """Split data into train/val/test sets"""
    print("\n" + "="*80)
    print("STEP 2: Split data")
    print("="*80)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    
    train_ratio = getattr(config.data, 'train_ratio', 0.8)
    val_ratio = getattr(config.data, 'val_ratio', 0.1)
    
    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def create_preprocessor(config, train_df: pd.DataFrame) -> PackageLifecyclePreprocessor:
    """Create and fit preprocessor on training data"""
    print("\n" + "="*80)
    print("STEP 3: Fit preprocessor")
    print("="*80)
    
    # Get distance file path
    distance_file = getattr(config.data, 'distance_file', 'data/location_distances_complete.csv')
    
    # Initialize preprocessor
    preprocessor = PackageLifecyclePreprocessor(
        config=config,
        distance_file_path=distance_file
    )
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Print feature dimensions
    feature_dims = preprocessor.get_feature_dimensions()
    print(f"\nFeature dimensions:")
    for key, value in feature_dims.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Print distance coverage
    dist_coverage = preprocessor.get_distance_coverage()
    print(f"\nDistance coverage:")
    for key, value in dist_coverage.items():
        print(f"  {key}: {value}")
    
    return preprocessor


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       preprocessor: PackageLifecyclePreprocessor, config) -> tuple:
    """Create PyG DataLoaders with cached preprocessing"""
    print("\n" + "="*80)
    print("STEP 4: Create datasets and dataloaders")
    print("="*80)
    
    # Get cache directory
    cache_dir = getattr(config.data, 'cache_dir', 'data/cache')
    
    num_workers = num_workers = max(1, os.cpu_count() - 1) 
    # Create datasets with caching - preprocessing happens once!
    print("\nCreating training dataset...")
    train_dataset = PackageLifecycleDataset(
        train_df, preprocessor, 
        return_labels=True,
    )
    
    print("\nCreating validation dataset...")
    val_dataset = PackageLifecycleDataset(
        val_df, preprocessor, 
        return_labels=True,
    )
    
    print("\nCreating test dataset...")
    test_dataset = PackageLifecycleDataset(
        test_df, preprocessor, 
        return_labels=True,
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Get batch size - can use larger batch since data loading is fast now
    batch_size = getattr(config.training, 'batch_size', 64)  # Increase batch size!
    
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch more batches
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"\nDataLoader settings:")
    print(f"  Batch size: {batch_size} (train), {batch_size*2} (val/test)")
    print(f"  Num workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def create_model(preprocessor: PackageLifecyclePreprocessor, config, device: torch.device) -> EventTimePredictor:
    """Create model from preprocessor and config"""
    print("\n" + "="*80)
    print("STEP 5: Initialize model")
    print("="*80)
    
    # Get vocab sizes
    vocab_sizes = preprocessor.get_vocab_sizes()
    
    # Create model config from preprocessor
    model_config = ModelConfig.from_preprocessor(
        preprocessor,
        hidden_dim=getattr(config.model, 'hidden_dim', 256),
        num_layers=getattr(config.model, 'num_layers', 20),
        num_heads=getattr(config.model, 'num_heads', 8),
        dropout=getattr(config.model, 'dropout', 0.1),
        embed_dim=getattr(config.model, 'embed_dim', 32),
        output_dim=getattr(config.model, 'output_dim', 1),
        use_edge_features=getattr(config.model, 'use_edge_features', True),
        use_global_attention=getattr(config.model, 'use_global_attention', True),
        use_positional_encoding=getattr(config.model, 'use_positional_encoding', True)
    )
    
    # Create model
    model = EventTimePredictor(model_config, vocab_sizes)
    model = model.to(device)
    
    # Print model info
    params = model.get_num_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    return model, model_config, vocab_sizes


def create_optimizer_and_scheduler(model: nn.Module, config, num_training_steps: int) -> tuple:
    """Create optimizer and learning rate scheduler"""
    # Optimizer
    learning_rate = getattr(config.training, 'learning_rate', 1e-4)
    weight_decay = getattr(config.training, 'weight_decay', 0.01)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    scheduler_type = getattr(config.training, 'scheduler_type', 'cosine')
    num_epochs = getattr(config.training, 'num_epochs', 100)
    
    scheduler = None
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
    elif scheduler_type == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=num_training_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
    
    print(f"\nOptimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Scheduler: {scheduler_type}")
    
    return optimizer, scheduler, scheduler_type


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, preprocessor: PackageLifecyclePreprocessor,
                scaler=None, scheduler=None, scheduler_type: str = None) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_samples = 0
    all_preds_scaled = []
    all_targets_scaled = []
    all_preds_hours = []
    all_targets_hours = []
    
    # Fixed progress bar - use position=0 and file=sys.stdout
    pbar = tqdm(
        loader, 
        desc='Training', 
        leave=False, 
        position=0,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5
    )
    
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            # Use the new torch.amp.autocast API
            with torch.amp.autocast('cuda'):
                predictions = model(batch)
                
                # Apply label mask to get valid predictions
                mask = batch.label_mask
                masked_preds = predictions[mask]
                masked_targets = batch.labels
                
                loss = criterion(masked_preds, masked_targets)
        else:
            predictions = model(batch)
            
            # Apply label mask
            mask = batch.label_mask
            masked_preds = predictions[mask]
            masked_targets = batch.labels
            
            loss = criterion(masked_preds, masked_targets)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Step scheduler if OneCycleLR (per-batch update)
        if scheduler is not None and scheduler_type == 'onecycle':
            scheduler.step()
        
        # Track metrics
        batch_size = masked_preds.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Store predictions (scaled)
        preds_np = masked_preds.detach().cpu().numpy()
        targets_np = masked_targets.detach().cpu().numpy()
        
        all_preds_scaled.append(preds_np)
        all_targets_scaled.append(targets_np)
        
        # Convert to hours
        preds_hours = preprocessor.inverse_transform_time(preds_np)
        targets_hours = preprocessor.inverse_transform_time(targets_np)
        
        all_preds_hours.append(preds_hours)
        all_targets_hours.append(targets_hours)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/total_samples:.4f}'
        })
    
    # Explicitly close progress bar
    pbar.close()
    
    # Compute epoch metrics
    avg_loss = total_loss / total_samples
    
    # Metrics on scaled values
    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)
    metrics_scaled = compute_metrics(all_preds_scaled, all_targets_scaled)
    
    # Metrics on hours
    all_preds_hours = np.concatenate(all_preds_hours, axis=0)
    all_targets_hours = np.concatenate(all_targets_hours, axis=0)
    metrics_hours = compute_metrics(all_preds_hours, all_targets_hours)
    
    metrics = {
        'loss': avg_loss,
        'mae_scaled': metrics_scaled.get('mae', 0),
        'rmse_scaled': metrics_scaled.get('rmse', 0),
        'mae_hours': metrics_hours.get('mae', 0),
        'rmse_hours': metrics_hours.get('rmse', 0),
        'mape': metrics_hours.get('mape', 0),
        'r2': metrics_hours.get('r2', 0),
        'num_samples': total_samples
    }
    
    return metrics


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, preprocessor: PackageLifecyclePreprocessor) -> dict:
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_preds_scaled = []
    all_targets_scaled = []
    all_preds_hours = []
    all_targets_hours = []
    
    # Fixed progress bar - use position=0 and file=sys.stdout
    pbar = tqdm(
        loader, 
        desc='Validating', 
        leave=False, 
        position=0,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5
    )
    
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        predictions = model(batch)
        
        # Apply label mask
        mask = batch.label_mask
        masked_preds = predictions[mask]
        masked_targets = batch.labels
        
        loss = criterion(masked_preds, masked_targets)
        
        # Track metrics
        batch_size = masked_preds.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Store predictions
        preds_np = masked_preds.cpu().numpy()
        targets_np = masked_targets.cpu().numpy()
        
        all_preds_scaled.append(preds_np)
        all_targets_scaled.append(targets_np)
        
        # Convert to hours
        preds_hours = preprocessor.inverse_transform_time(preds_np)
        targets_hours = preprocessor.inverse_transform_time(targets_np)
        
        all_preds_hours.append(preds_hours)
        all_targets_hours.append(targets_hours)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Explicitly close progress bar
    pbar.close()
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    
    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)
    metrics_scaled = compute_metrics(all_preds_scaled, all_targets_scaled)
    
    all_preds_hours = np.concatenate(all_preds_hours, axis=0)
    all_targets_hours = np.concatenate(all_targets_hours, axis=0)
    metrics_hours = compute_metrics(all_preds_hours, all_targets_hours)
    
    metrics = {
        'loss': avg_loss,
        'mae_scaled': metrics_scaled.get('mae', 0),
        'rmse_scaled': metrics_scaled.get('rmse', 0),
        'mae_hours': metrics_hours.get('mae', 0),
        'rmse_hours': metrics_hours.get('rmse', 0),
        'mape': metrics_hours.get('mape', 0),
        'r2': metrics_hours.get('r2', 0),
        'num_samples': total_samples
    }
    
    return metrics


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, epoch: int, metrics: dict,
                    model_config: ModelConfig, vocab_sizes: dict,
                    save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config.to_dict(),
        'vocab_sizes': vocab_sizes,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler = None, device: torch.device = None) -> dict:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def main():
    """Main training function"""
    
    # ========================================
    # Setup
    # ========================================
    
    # Load configuration
    config = Config()
    set_seed(config.seed)
    
    # Create save directories with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(getattr(config.training, 'save_dir', 'checkpoints'), timestamp)
    log_dir = os.path.join(getattr(config.training, 'log_dir', 'logs'), timestamp)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*80)
    print("Package Event Time Prediction - Training")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Seed: {config.seed}")
    
    # Device
    device_str = getattr(config.training, 'device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ========================================
    # Data Loading and Preprocessing
    # ========================================
    
    # Load data
    df = load_data(config)
    
    # Split data
    train_df, val_df, test_df = split_data(df, config)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(config, train_df)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, preprocessor, config
    )
    
    # ========================================
    # Model Setup
    # ========================================
    
    # Create model
    model, model_config, vocab_sizes = create_model(preprocessor, config, device)
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader)
    optimizer, scheduler, scheduler_type = create_optimizer_and_scheduler(
        model, config, num_training_steps
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Early stopping
    patience = getattr(config.training, 'patience', 15)
    min_delta = getattr(config.training, 'min_delta', 1e-4)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Mixed precision scaler - use the new API
    use_amp = getattr(config.training, 'use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    if use_amp:
        print("Using mixed precision training (AMP)")
    
    # ========================================
    # Save Configuration
    # ========================================
    
    # Save config
    config_dict = {
        'timestamp': timestamp,
        'seed': config.seed,
        'model_config': model_config.to_dict(),
        'vocab_sizes': vocab_sizes,
        'feature_dims': preprocessor.get_feature_dimensions(),
        'training': {
            'batch_size': getattr(config.training, 'batch_size', 32),
            'learning_rate': getattr(config.training, 'learning_rate', 1e-4),
            'weight_decay': getattr(config.training, 'weight_decay', 0.01),
            'num_epochs': getattr(config.training, 'num_epochs', 100),
            'scheduler_type': scheduler_type,
            'patience': patience,
            'min_delta': min_delta,
            'use_amp': use_amp,
        },
        'data': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        }
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Save preprocessor
    preprocessor.save(os.path.join(save_dir, 'preprocessor.pkl'))
    
    # ========================================
    # Training Loop
    # ========================================
    
    print("\n" + "="*80)
    print("STEP 6: Training")
    print("="*80)
    
    num_epochs = getattr(config.training, 'num_epochs', 100)
    save_every = getattr(config.training, 'save_every', 10)
    
    best_val_loss = float('inf')
    best_val_mae_hours = float('inf')
    best_epoch = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae_hours': [],
        'val_mae_hours': [],
        'train_rmse_hours': [],
        'val_rmse_hours': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            preprocessor=preprocessor,
            scaler=scaler,
            scheduler=scheduler if scheduler_type == 'onecycle' else None,
            scheduler_type=scheduler_type
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            preprocessor=preprocessor
        )
        
        # Update scheduler (except OneCycleLR which updates per step)
        if scheduler is not None and scheduler_type != 'onecycle':
            scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/mae_hours', train_metrics['mae_hours'], epoch)
        writer.add_scalar('train/rmse_hours', train_metrics['rmse_hours'], epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/mae_hours', val_metrics['mae_hours'], epoch)
        writer.add_scalar('val/rmse_hours', val_metrics['rmse_hours'], epoch)
        writer.add_scalar('val/r2', val_metrics['r2'], epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Update history
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['train_mae_hours'].append(train_metrics['mae_hours'])
        training_history['val_mae_hours'].append(val_metrics['mae_hours'])
        training_history['train_rmse_hours'].append(train_metrics['rmse_hours'])
        training_history['val_rmse_hours'].append(val_metrics['rmse_hours'])
        training_history['val_r2'].append(val_metrics['r2'])
        training_history['learning_rate'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"MAE: {train_metrics['mae_hours']:.2f}h, "
              f"RMSE: {train_metrics['rmse_hours']:.2f}h")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"MAE: {val_metrics['mae_hours']:.2f}h, "
              f"RMSE: {val_metrics['rmse_hours']:.2f}h, "
              f"R²: {val_metrics['r2']:.4f}")
        print(f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_mae_hours = val_metrics['mae_hours']
            best_epoch = epoch
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'val_loss': val_metrics['loss'],
                    'val_mae_hours': val_metrics['mae_hours'],
                    'val_rmse_hours': val_metrics['rmse_hours'],
                    'val_r2': val_metrics['r2']
                },
                model_config=model_config,
                vocab_sizes=vocab_sizes,
                save_path=os.path.join(save_dir, 'best_model.pt')
            )
            print(f"✓ New best model saved (val_loss: {best_val_loss:.4f}, MAE: {best_val_mae_hours:.2f}h)")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'val_loss': val_metrics['loss'],
                    'val_mae_hours': val_metrics['mae_hours']
                },
                model_config=model_config,
                vocab_sizes=vocab_sizes,
                save_path=os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
            print(f"✓ Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\n{'='*50}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best epoch: {best_epoch+1} with val_loss: {best_val_loss:.4f}")
            print(f"{'='*50}")
            break
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={
            'val_loss': val_metrics['loss'],
            'val_mae_hours': val_metrics['mae_hours']
        },
        model_config=model_config,
        vocab_sizes=vocab_sizes,
        save_path=os.path.join(save_dir, 'final_model.pt')
    )
    
    # ========================================
    # Testing
    # ========================================
    
    print("\n" + "="*80)
    print("STEP 7: Testing on best model")
    print("="*80)
    
    # Load best model
    checkpoint = load_checkpoint(
        os.path.join(save_dir, 'best_model.pt'),
        model=model,
        device=device
    )
    
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test
    test_metrics = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        preprocessor=preprocessor
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE (scaled): {test_metrics['mae_scaled']:.4f}")
    print(f"  RMSE (scaled): {test_metrics['rmse_scaled']:.4f}")
    print(f"  MAE (hours): {test_metrics['mae_hours']:.2f}")
    print(f"  RMSE (hours): {test_metrics['rmse_hours']:.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save test results
    test_results = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'best_val_mae_hours': float(best_val_mae_hours),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'training_epochs': epoch + 1,
        'early_stopped': early_stopping.counter >= early_stopping.patience
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # ========================================
    # Cleanup
    # ========================================
    
    writer.close()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val MAE: {best_val_mae_hours:.2f} hours")
    print(f"Test MAE: {test_metrics['mae_hours']:.2f} hours")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"\nResults saved to: {save_dir}")
    print("="*80)
    
    return {
        'save_dir': save_dir,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics,
        'training_history': training_history
    }



if __name__ == '__main__':
    results = main()