import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    r2 = r2_score(targets, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0001):
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