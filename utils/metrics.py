import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(preds, targets, preprocessor=None):
    """
    Compute regression metrics
    
    Args:
        preds: predictions (scaled if preprocessor is provided)
        targets: ground truth (scaled if preprocessor is provided)
        preprocessor: optional preprocessor to inverse transform values
    
    Returns:
        dict with metrics (includes both scaled and inverted versions if preprocessor provided)
    """
    # Ensure same shape
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    
    # Compute scaled metrics
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    
    # R² score
    r2 = r2_score(targets, preds)
    
    # MAPE (avoid division by zero)
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }
    
    # ✅ If preprocessor provided, compute inverted (original scale) metrics
    if preprocessor is not None:
        # Inverse transform predictions and targets
        preds_inv = preprocessor.inverse_transform_time(preds.reshape(-1, 1)).flatten()
        targets_inv = preprocessor.inverse_transform_time(targets.reshape(-1, 1)).flatten()
        
        # Compute metrics in original scale (hours)
        mae_hours = mean_absolute_error(targets_inv, preds_inv)
        rmse_hours = np.sqrt(mean_squared_error(targets_inv, preds_inv))
        
        # Add to metrics
        metrics['mae_hours'] = float(mae_hours)
        metrics['rmse_hours'] = float(rmse_hours)
    
    return metrics


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0.0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return False