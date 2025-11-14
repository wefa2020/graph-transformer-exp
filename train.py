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
from utils.metrics import compute_metrics, EarlyStopping

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
    #    start_date='2025-11-01T08:00:00Z',
    #    end_date='2025-11-08T08:00:00Z'
    #)
    
    print("Execute this query in Neptune and save results to 'package_lifecycles.json'")
    print("\nQuery:")
    #print(query)
    
    # Load from JSON (after manual extraction)
    print("\n" + "="*80)
    print("STEP 2: Load extracted data")
    print("="*80)
    
    df = extractor.load_from_json('package_lifecycles.json') 
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

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
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
                                # Only compute loss for nodes with labels
                
                loss = criterion(masked_preds, batch.y)  # Exclude last node
        else:
            preds = model(batch)
            # Use label mask to select only nodes with labels
            if hasattr(batch, 'label_mask'):
                masked_preds = preds[batch.label_mask]
            else:
                # Fallback: manually exclude last node of each graph
                masked_preds = exclude_last_nodes(preds, batch.batch)
                                # Only compute loss for nodes with labels
                
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
        # ✅ Collect AFTER masking (both same size now)
        all_preds.append(masked_preds.detach().cpu())
        all_targets.append(batch.y.detach().cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device):
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
                                # Only compute loss for nodes with labels
                
        loss = criterion(masked_preds, batch.y)
        
        # Track metrics
        total_loss += loss.item()
        all_preds.append(masked_preds.detach().cpu())
        all_targets.append(batch.y.detach().cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    metrics = compute_metrics(all_preds, all_targets)
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
    
    # Early stopping
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
    
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if config.training.scheduler_type != 'onecycle':
            scheduler.step()
        
        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'preprocessor': preprocessor
            }, os.path.join(config.training.save_dir, 'best_model.pt'))
            print("✓ Saved best model")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'preprocessor': preprocessor
            }, os.path.join(config.training.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test on best model
    print("\n" + "="*80)
    print("STEP 7: Testing")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(config.training.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    writer.close()
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

if __name__ == '__main__':
    main()