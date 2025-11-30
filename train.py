import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.neptune_extractor import NeptuneDataExtractor
from data.data_preprocessor import PackageLifecyclePreprocessor
from data.dataset import PackageLifecycleDataset, collate_fn
from models.event_predictor import EventTimePredictor
from utils.metrics import compute_metrics, compute_selection_score, EarlyStopping
from utils.package_filter import PackageEventValidator


def exclude_last_nodes(preds, batch_vector):
    """
    Exclude the last node from each graph in the batch.
    
    Args:
        preds: predictions for all nodes (shape: [total_nodes, ...])
        batch_vector: indicates which graph each node belongs to
    
    Returns:
        predictions excluding last node of each graph
    """
    # Find the last node index for each graph
    num_graphs = batch_vector.max().item() + 1
    mask = torch.ones(len(preds), dtype=torch.bool, device=preds.device)
    
    for graph_id in range(num_graphs):
        # Find all nodes belonging to this graph
        node_indices = (batch_vector == graph_id).nonzero(as_tuple=True)[0]
        # Mark the last node as False
        if len(node_indices) > 0:
            last_node_idx = node_indices[-1]
            mask[last_node_idx] = False
    
    return preds[mask]


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dataloaders(config):
    """Create train/val/test dataloaders"""
    
    print("Loading data from Neptune...")
    
    # Option 1: Extract from Neptune (you need to run the query manually)
    extractor = NeptuneDataExtractor(config.neptune.endpoint)
    
    # For demonstration, assume data is already extracted to JSON
    # In practice, you would execute the Cypher query and save results
    print("\n" + "="*80)
    print("STEP 1: Extract data from Neptune")
    print("="*80)
    
    #query = extractor.extract_lifecycles(
    #    start_date='2025-11-01T00:00:00Z',
    #    end_date='2025-11-08T23:59:59Z'
    #)
    
    print("Execute this query in Neptune and save results to 'package_lifecycles.json'")
    print("\nQuery:")
    #print(query)
    
    # Load from JSON (after manual extraction)
    print("\n" + "="*80)
    print("STEP 2: Load extracted data")
    print("="*80)
    
    df = extractor.load_from_json('source.json') 
    print(f"Loaded {len(df)} package lifecycles")
    
    # Split data
    train_size = int(config.data.train_ratio * len(df))
    val_size = int(config.data.val_ratio * len(df))
    test_size = len(df) - train_size - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Fit preprocessor on training data
    print("\n" + "="*80)
    print("STEP 3: Fit preprocessor")
    print("="*80)
    
    preprocessor = PackageLifecyclePreprocessor(config)
    preprocessor.fit(df)
    
    # Determine input dimension
    sample_processed = preprocessor.process_lifecycle(train_df.iloc[0].to_dict())
    node_feature_dim = sample_processed['node_features'].shape[1]
    edge_feature_dim = sample_processed['edge_features'].shape[1] if len(sample_processed['edge_features']) > 0 else 0
    
    config.model.node_feature_dim = node_feature_dim
    config.model.edge_feature_dim = edge_feature_dim
    
    print(f"Node feature dim: {node_feature_dim}")
    print(f"Edge feature dim: {edge_feature_dim}")
    
    # Create datasets
    print("\n" + "="*80)
    print("STEP 4: Create datasets")
    print("="*80)
    
    train_dataset = PackageLifecycleDataset(train_df, preprocessor, return_labels=True)
    val_dataset = PackageLifecycleDataset(val_df, preprocessor, return_labels=True)
    test_dataset = PackageLifecycleDataset(test_df, preprocessor, return_labels=True)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, preprocessor


def train_epoch(model, loader, optimizer, criterion, device, preprocessor, scaler=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc='Training')
    
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(batch)
                # Use label mask to select only nodes with labels
                if hasattr(batch, 'label_mask'):
                    masked_preds = preds[batch.label_mask]
                else:
                    # Fallback: manually exclude last node of each graph
                    masked_preds = exclude_last_nodes(preds, batch.batch)
                
                loss = criterion(masked_preds, batch.y)
        else:
            preds = model(batch)
            # Use label mask to select only nodes with labels
            if hasattr(batch, 'label_mask'):
                masked_preds = preds[batch.label_mask]
            else:
                # Fallback: manually exclude last node of each graph
                masked_preds = exclude_last_nodes(preds, batch.batch)
            
            loss = criterion(masked_preds, batch.y)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        # Collect AFTER masking (both same size now)
        all_preds.append(masked_preds.detach().cpu())
        all_targets.append(batch.y.detach().cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics with inverse transform
    metrics = compute_metrics(all_preds, all_targets, preprocessor=preprocessor)
    metrics['loss'] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc='Validating')
    
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        preds = model(batch)
        # Use label mask to select only nodes with labels
        if hasattr(batch, 'label_mask'):
            masked_preds = preds[batch.label_mask]
        else:
            # Fallback: manually exclude last node of each graph
            masked_preds = exclude_last_nodes(preds, batch.batch)
        
        loss = criterion(masked_preds, batch.y)
        
        # Track metrics
        total_loss += loss.item()
        all_preds.append(masked_preds.detach().cpu())
        all_targets.append(batch.y.detach().cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics with inverse transform
    metrics = compute_metrics(all_preds, all_targets, preprocessor=preprocessor)
    metrics['loss'] = avg_loss
    
    return metrics


def main():
    # Load configuration
    config = Config()
    set_seed(config.seed)
    
    # Create save directories
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, preprocessor = create_dataloaders(config)
    
    # Create model
    print("\n" + "="*80)
    print("STEP 5: Initialize model")
    print("="*80)
    
    device = torch.device(config.training.device)
    model = EventTimePredictor(config).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler
    if config.training.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    elif config.training.scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            epochs=config.training.num_epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = None
    
    # Early stopping (based on composite score)
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # Tensorboard
    writer = SummaryWriter(config.training.log_dir)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    print("\n" + "="*80)
    print("STEP 6: Training")
    print("="*80)
    
    best_score = float('inf')
    
    for epoch in range(config.training.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print('='*80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, preprocessor, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, preprocessor)
        
        # Update scheduler
        if scheduler is not None:
            if config.training.scheduler_type == 'onecycle':
                # OneCycleLR updates every batch, so skip here
                pass
            else:
                scheduler.step()
        
        # Compute composite score for model selection (using MAPE)
        train_score = compute_selection_score(train_metrics, mape_weight=0.4)
        val_score = compute_selection_score(val_metrics, mape_weight=0.4)
        
        # Log all metrics to tensorboard
        for key, value in train_metrics.items():
            # With:
            if value is not None and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                writer.add_scalar(f'train/{key}', value, epoch)
            else:
                print(f"Warning: Skipping logging for train/{key} due to invalid value: {value}")
            #writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            # With:
            if value is not None and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                writer.add_scalar(f'val/{key}', value, epoch)
            else:
                print(f"Warning: Skipping logging for train/{key} due to invalid value: {value}")
            
            #writer.add_scalar(f'val/{key}', value, epoch)
        
        writer.add_scalar('train/composite_score', train_score, epoch)
        writer.add_scalar('val/composite_score', val_score, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"\nTrain Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  MAE: {train_metrics['mae_hours']:.2f} hours")
        print(f"  RMSE: {train_metrics['rmse_hours']:.2f} hours")
        print(f"  MAPE: {train_metrics['mape']:.2f}%")
        print(f"  R²: {train_metrics['r2']:.4f}")
        print(f"  Bias: {train_metrics['mean_signed_error_hours']:+.2f} hours "
              f"(Over: {train_metrics['mae_hours_positive']:.2f}h, Under: {train_metrics['mae_hours_negative']:.2f}h)")
        print(f"  Composite Score: {train_score:.2f}")
        
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  MAE: {val_metrics['mae_hours']:.2f} hours")
        print(f"  RMSE: {val_metrics['rmse_hours']:.2f} hours")
        print(f"  MAPE: {val_metrics['mape']:.2f}%")
        print(f"  R²: {val_metrics['r2']:.4f}")
        print(f"  Bias: {val_metrics['mean_signed_error_hours']:+.2f} hours "
              f"(Over: {val_metrics['mae_hours_positive']:.2f}h, Under: {val_metrics['mae_hours_negative']:.2f}h)")
        print(f"  Composite Score: {val_score:.2f} ⭐")
        
        # Save best model based on composite score
        if val_score < best_score:
            improvement = ((best_score - val_score) / best_score * 100) if best_score != float('inf') else 0
            best_score = val_score
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'composite_score': val_score,
                'config': config,
                'preprocessor': preprocessor
            }, os.path.join(config.training.save_dir, 'best_model.pt'))
            
            print(f"\n✅ Saved best model (score: {best_score:.2f}, improvement: {improvement:+.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'composite_score': val_score,
                'config': config,
                'preprocessor': preprocessor
            }, os.path.join(config.training.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
        
        # Early stopping based on composite score
        if early_stopping(val_score):
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test on best model
    print("\n" + "="*80)
    print("STEP 7: Final Testing")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(config.training.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device, preprocessor)
    test_score = compute_selection_score(test_metrics, mape_weight=0.4)
    
    print(f"\n🎯 Test Results:")
    print("="*80)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"MAE: {test_metrics['mae_hours']:.2f} hours")
    print(f"RMSE: {test_metrics['rmse_hours']:.2f} hours")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"Bias: {test_metrics['mean_signed_error_hours']:+.2f} hours")
    print(f"  Over-predictions MAE: {test_metrics['mae_hours_positive']:.2f} hours")
    print(f"  Under-predictions MAE: {test_metrics['mae_hours_negative']:.2f} hours")
    print(f"\n⭐ Composite Score: {test_score:.2f} (MAE + MAPE*0.4)")
    print("="*80)
    
    # Log test metrics to tensorboard
    for key, value in test_metrics.items():
        writer.add_scalar(f'test/{key}', value, 0)
    writer.add_scalar('test/composite_score', test_score, 0)
    
    writer.close()
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"Best model saved at: {os.path.join(config.training.save_dir, 'best_model.pt')}")
    print(f"TensorBoard logs at: {config.training.log_dir}")
    print("="*80)


if __name__ == '__main__':
    main()