import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import os


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, 
                    prefix: str = '') -> Dict[str, float]:
    """
    Compute comprehensive regression metrics
    
    Args:
        predictions: Predicted values [N, 1] or [N]
        targets: Target values [N, 1] or [N]
        prefix: Optional prefix for metric names
    
    Returns:
        Dict with metrics: mae, rmse, mse, mape, smape, r2, explained_variance
    """
    # Flatten arrays
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    
    # Ensure same length
    assert len(preds) == len(targs), f"Length mismatch: {len(preds)} vs {len(targs)}"
    
    n_samples = len(preds)
    
    if n_samples == 0:
        return {f'{prefix}mae': 0.0, f'{prefix}rmse': 0.0, f'{prefix}r2': 0.0}
    
    # Errors
    errors = preds - targs
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # MAE (Mean Absolute Error)
    mae = np.mean(abs_errors)
    
    # MSE (Mean Squared Error)
    mse = np.mean(squared_errors)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    mask = np.abs(targs) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((targs[mask] - preds[mask]) / targs[mask])) * 100
    else:
        mape = 0.0
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(preds) + np.abs(targs)) / 2
    mask_smape = denominator > 1e-8
    if mask_smape.sum() > 0:
        smape = np.mean(abs_errors[mask_smape] / denominator[mask_smape]) * 100
    else:
        smape = 0.0
    
    # R² Score (Coefficient of Determination)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((targs - np.mean(targs)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 0.0
    
    # Explained Variance Score
    var_res = np.var(errors)
    var_targ = np.var(targs)
    explained_variance = 1 - (var_res / (var_targ + 1e-8)) if var_targ > 1e-8 else 0.0
    
    # Max Error
    max_error = np.max(abs_errors)
    
    # Median Absolute Error
    median_ae = np.median(abs_errors)
    
    # Percentiles
    p50 = np.percentile(abs_errors, 50)
    p90 = np.percentile(abs_errors, 90)
    p95 = np.percentile(abs_errors, 95)
    p99 = np.percentile(abs_errors, 99)
    
    # Build result dict
    result = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'smape': float(smape),
        'r2': float(r2),
        'explained_variance': float(explained_variance),
        'max_error': float(max_error),
        'median_ae': float(median_ae),
        'p50': float(p50),
        'p90': float(p90),
        'p95': float(p95),
        'p99': float(p99),
        'n_samples': n_samples
    }
    
    # Add prefix if specified
    if prefix:
        result = {f'{prefix}{k}': v for k, v in result.items()}
    
    return result


def compute_metrics_by_range(predictions: np.ndarray, targets: np.ndarray,
                             ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for different target value ranges
    
    Args:
        predictions: Predicted values
        targets: Target values
        ranges: List of (min, max) tuples defining ranges. Default: [(0,1), (1,6), (6,24), (24,inf)]
    
    Returns:
        Dict mapping range names to metrics
    """
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    
    if ranges is None:
        ranges = [
            (0, 1),      # 0-1 hours
            (1, 6),      # 1-6 hours
            (6, 24),     # 6-24 hours
            (24, 72),    # 1-3 days
            (72, float('inf'))  # 3+ days
        ]
    
    results = {}
    
    for min_val, max_val in ranges:
        mask = (targs >= min_val) & (targs < max_val)
        
        if mask.sum() > 0:
            range_name = f'{min_val:.0f}-{max_val:.0f}h' if max_val < float('inf') else f'{min_val:.0f}+h'
            results[range_name] = compute_metrics(preds[mask], targs[mask])
            results[range_name]['count'] = int(mask.sum())
            results[range_name]['percentage'] = float(mask.sum() / len(targs) * 100)
    
    return results


def compute_metrics_by_category(predictions: np.ndarray, targets: np.ndarray,
                                categories: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics grouped by category
    
    Args:
        predictions: Predicted values
        targets: Target values
        categories: Category labels for each sample
    
    Returns:
        Dict mapping category names to metrics
    """
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    cats = np.asarray(categories).flatten()
    
    unique_cats = np.unique(cats)
    results = {}
    
    for cat in unique_cats:
        mask = cats == cat
        if mask.sum() > 0:
            cat_name = str(cat)
            results[cat_name] = compute_metrics(preds[mask], targs[mask])
            results[cat_name]['count'] = int(mask.sum())
    
    return results


def compute_directional_accuracy(predictions: np.ndarray, targets: np.ndarray,
                                 threshold: float = 0.0) -> Dict[str, float]:
    """
    Compute directional/sign accuracy metrics
    
    Args:
        predictions: Predicted values
        targets: Target values
        threshold: Threshold for considering prediction correct
    
    Returns:
        Dict with directional accuracy metrics
    """
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    
    # Within threshold accuracy
    within_threshold = np.abs(preds - targs) <= threshold
    threshold_accuracy = np.mean(within_threshold) * 100
    
    # Underestimate vs overestimate
    underestimate = preds < targs
    overestimate = preds > targs
    exact = preds == targs
    
    return {
        'threshold_accuracy': float(threshold_accuracy),
        'underestimate_rate': float(np.mean(underestimate) * 100),
        'overestimate_rate': float(np.mean(overestimate) * 100),
        'exact_rate': float(np.mean(exact) * 100),
        'mean_bias': float(np.mean(preds - targs)),  # Positive = overestimate
    }


def compute_time_based_metrics(predictions: np.ndarray, targets: np.ndarray,
                               tolerance_hours: List[float] = None) -> Dict[str, float]:
    """
    Compute time-based accuracy metrics
    
    Args:
        predictions: Predicted values in hours
        targets: Target values in hours
        tolerance_hours: List of tolerance thresholds in hours
    
    Returns:
        Dict with accuracy at different tolerances
    """
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    
    if tolerance_hours is None:
        tolerance_hours = [0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0]
    
    abs_errors = np.abs(preds - targs)
    
    results = {}
    for tol in tolerance_hours:
        within_tol = abs_errors <= tol
        accuracy = np.mean(within_tol) * 100
        results[f'accuracy_within_{tol}h'] = float(accuracy)
    
    return results


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    
    Supports both minimization (loss) and maximization (accuracy, R²) modes
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 mode: str = 'min', verbose: bool = False):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.should_stop = False
        self.history = []
    
    def __call__(self, value: float, epoch: int = None) -> bool:
        """
        Check if training should stop
        
        Args:
            value: Current metric value
            epoch: Current epoch number (optional)
        
        Returns:
            True if training should stop
        """
        self.history.append(value)
        
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch or 0
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            if self.verbose:
                print(f"  EarlyStopping: Metric improved from {self.best_value:.6f} to {value:.6f}")
            self.best_value = value
            self.best_epoch = epoch or len(self.history) - 1
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: No improvement. Counter: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"  EarlyStopping: Triggered! Best value: {self.best_value:.6f} at epoch {self.best_epoch}")
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.should_stop = False
        self.history = []
    
    def state_dict(self) -> Dict:
        """Get state for serialization"""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'should_stop': self.should_stop,
            'history': self.history
        }
    
    def load_state_dict(self, state: Dict):
        """Load state from dict"""
        self.patience = state.get('patience', self.patience)
        self.min_delta = state.get('min_delta', self.min_delta)
        self.mode = state.get('mode', self.mode)
        self.counter = state.get('counter', 0)
        self.best_value = state.get('best_value', None)
        self.best_epoch = state.get('best_epoch', None)
        self.should_stop = state.get('should_stop', False)
        self.history = state.get('history', [])


@dataclass
class MetricTracker:
    """Track metrics over training epochs"""
    
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def update(self, metric_dict: Dict[str, float], prefix: str = ''):
        """Add metrics for current epoch"""
        for key, value in metric_dict.items():
            full_key = f'{prefix}{key}' if prefix else key
            if full_key not in self.metrics:
                self.metrics[full_key] = []
            self.metrics[full_key].append(float(value))
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[int, float]:
        """Get best value and epoch for a metric"""
        if metric_name not in self.metrics:
            return -1, float('inf') if mode == 'min' else float('-inf')
        
        values = self.metrics[metric_name]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return int(best_idx), float(values[best_idx])
    
    def get_last(self, metric_name: str) -> Optional[float]:
        """Get last value for a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        return self.metrics[metric_name][-1]
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get full history for a metric"""
        return self.metrics.get(metric_name, [])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[key] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'last': float(values[-1]),
                    'best_epoch_min': int(np.argmin(values)),
                    'best_epoch_max': int(np.argmax(values)),
                }
        return summary
    
    def save(self, path: str):
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from JSON file"""
        with open(path, 'r') as f:
            self.metrics = json.load(f)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.metrics)


class RunningAverage:
    """Compute running average of a metric"""
    
    def __init__(self):
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        """Update with new value(s)"""
        self.total += value * n
        self.count += n
    
    @property
    def average(self) -> float:
        """Get current average"""
        return self.total / self.count if self.count > 0 else 0.0
    
    def reset(self):
        """Reset the running average"""
        self.total = 0.0
        self.count = 0


class ExponentialMovingAverage:
    """Compute exponential moving average of a metric"""
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more weight on recent values
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float):
        """Update with new value"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
    
    @property
    def average(self) -> float:
        """Get current EMA value"""
        return self.value if self.value is not None else 0.0
    
    def reset(self):
        """Reset the EMA"""
        self.value = None


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dict of metric name to value
        title: Table title
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    max_key_len = max(len(k) for k in metrics.keys())
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                print(f"  {key:<{max_key_len}}: {value:.4e}")
            else:
                print(f"  {key:<{max_key_len}}: {value:.4f}")
        else:
            print(f"  {key:<{max_key_len}}: {value}")
    
    print(f"{'='*50}\n")


def format_metrics_string(metrics: Dict[str, float], keys: List[str] = None,
                          precision: int = 4) -> str:
    """
    Format metrics as a single-line string
    
    Args:
        metrics: Dict of metrics
        keys: Specific keys to include (default: all)
        precision: Decimal precision
    
    Returns:
        Formatted string like "mae=1.23, rmse=2.34, r2=0.89"
    """
    if keys is None:
        keys = list(metrics.keys())
    
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.{precision}f}")
            else:
                parts.append(f"{key}={value}")
    
    return ", ".join(parts)


def compare_metrics(metrics1: Dict[str, float], metrics2: Dict[str, float],
                    name1: str = "Model 1", name2: str = "Model 2") -> Dict[str, Dict]:
    """
    Compare two sets of metrics
    
    Args:
        metrics1: First set of metrics
        metrics2: Second set of metrics
        name1: Name for first model
        name2: Name for second model
    
    Returns:
        Dict with comparison results
    """
    all_keys = set(metrics1.keys()) | set(metrics2.keys())
    
    comparison = {}
    for key in all_keys:
        v1 = metrics1.get(key, None)
        v2 = metrics2.get(key, None)
        
        if v1 is not None and v2 is not None:
            diff = v2 - v1
            pct_change = (diff / abs(v1) * 100) if abs(v1) > 1e-8 else 0.0
            
            comparison[key] = {
                name1: v1,
                name2: v2,
                'difference': diff,
                'pct_change': pct_change,
                'better': name2 if diff < 0 else name1  # Assumes lower is better
            }
    
    return comparison


def save_metrics_report(metrics: Dict, filepath: str, include_timestamp: bool = True):
    """
    Save metrics report to file
    
    Args:
        metrics: Metrics dict
        filepath: Output file path
        include_timestamp: Whether to include timestamp
    """
    from datetime import datetime
    
    report = {
        'metrics': metrics
    }
    
    if include_timestamp:
        report['timestamp'] = datetime.now().isoformat()
    
    # Determine format from extension
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif ext == '.txt':
        with open(filepath, 'w') as f:
            f.write(f"Metrics Report\n")
            if include_timestamp:
                f.write(f"Generated: {report['timestamp']}\n")
            f.write("="*50 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    else:
        # Default to JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)