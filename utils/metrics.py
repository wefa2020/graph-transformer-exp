import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_pred, y_true, preprocessor=None, package_data=None):
    """
    Compute comprehensive metrics for transit time predictions
    
    Args:
        y_pred: Predicted transit times (scaled or unscaled), shape (n_samples, 1) or (n_samples,)
        y_true: True transit times (scaled or unscaled), shape (n_samples, 1) or (n_samples,)
        preprocessor: Preprocessor with inverse_transform_time method
        package_data: Optional (not used for transit time prediction)
    
    Returns:
        Dictionary of metrics (None values are used for unavailable metrics)
    """
    # Ensure arrays are 1D
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    
    # Check for empty arrays
    if len(y_pred) == 0 or len(y_true) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'r2': 0.0,
            'mae_hours': 0.0,
            'rmse_hours': 0.0,
            'mape': None,
            'median_ape': None,
            'mape_samples': 0,
            'median_error': 0.0,
            'mean_signed_error_hours': 0.0,
        }
    
    metrics = {}
    
    # Basic metrics (scaled)
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # R² score (handle edge cases)
    try:
        r2 = r2_score(y_true, y_pred)
        if np.isnan(r2) or np.isinf(r2):
            metrics['r2'] = 0.0
        else:
            metrics['r2'] = float(r2)
    except:
        metrics['r2'] = 0.0
    
    # Directional metrics (detect cancellation/bias)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    metrics['mae_positive'] = float(np.mean(abs_errors[errors > 0])) if (errors > 0).any() else 0.0
    metrics['mae_negative'] = float(np.mean(abs_errors[errors < 0])) if (errors < 0).any() else 0.0
    metrics['mean_signed_error'] = float(np.mean(errors))
    
    # Count directional errors
    metrics['n_positive_errors'] = int((errors > 0).sum())
    metrics['n_negative_errors'] = int((errors < 0).sum())
    
    # Inverse transform to hours (these are TRANSIT TIMES in hours)
    if preprocessor is not None and hasattr(preprocessor, 'inverse_transform_time'):
        try:
            y_pred_hours = preprocessor.inverse_transform_time(y_pred.reshape(-1, 1)).flatten()
            y_true_hours = preprocessor.inverse_transform_time(y_true.reshape(-1, 1)).flatten()
            
            errors_hours = y_pred_hours - y_true_hours
            abs_errors_hours = np.abs(errors_hours)
            
            # Transit time metrics in hours
            metrics['mae_hours'] = float(np.mean(abs_errors_hours))
            metrics['rmse_hours'] = float(np.sqrt(np.mean(errors_hours**2)))
            
            # Directional in hours
            metrics['mae_hours_positive'] = float(np.mean(abs_errors_hours[errors_hours > 0])) if (errors_hours > 0).any() else 0.0
            metrics['mae_hours_negative'] = float(np.mean(abs_errors_hours[errors_hours < 0])) if (errors_hours < 0).any() else 0.0
            metrics['mean_signed_error_hours'] = float(np.mean(errors_hours))
            
            # Distribution metrics
            metrics['median_error'] = float(np.median(abs_errors_hours))
            metrics['p25_error'] = float(np.percentile(abs_errors_hours, 25))
            metrics['p75_error'] = float(np.percentile(abs_errors_hours, 75))
            metrics['p90_error'] = float(np.percentile(abs_errors_hours, 90))
            metrics['p95_error'] = float(np.percentile(abs_errors_hours, 95))
            metrics['p99_error'] = float(np.percentile(abs_errors_hours, 99))
            
            # Standard deviation
            metrics['std_error_hours'] = float(np.std(errors_hours))
            
            # ================================================================
            # MAPE: Calculate on TRANSIT TIMES
            # ================================================================
            mape_result = _calculate_mape_from_transit_times(y_pred_hours, y_true_hours)
            metrics.update(mape_result)
            
        except Exception as e:
            print(f"Warning: Error in inverse transform: {e}")
            # Set defaults if transformation fails
            metrics['mae_hours'] = metrics['mae']
            metrics['rmse_hours'] = metrics['rmse']
            metrics['median_error'] = float(np.median(abs_errors))
            metrics['mean_signed_error_hours'] = metrics['mean_signed_error']
            metrics['mape'] = None
            metrics['median_ape'] = None
            metrics['mape_samples'] = 0
    else:
        # No preprocessor - use scaled values as proxy
        metrics['mae_hours'] = metrics['mae']
        metrics['rmse_hours'] = metrics['rmse']
        metrics['median_error'] = float(np.median(abs_errors))
        metrics['mean_signed_error_hours'] = metrics['mean_signed_error']
        
        # Calculate MAPE on scaled values
        mape_result = _calculate_mape_from_transit_times(y_pred, y_true)
        metrics.update(mape_result)
    
    return metrics


def _calculate_mape_from_transit_times(y_pred_hours, y_true_hours, min_threshold=0.1):
    """
    Calculate MAPE on transit time predictions
    MAPE = mean(|predicted_time - actual_time| / |actual_time|) * 100
    
    Only includes samples where actual_time > min_threshold to avoid division by near-zero
    
    Args:
        y_pred_hours: Predicted transit times in hours
        y_true_hours: Actual transit times in hours
        min_threshold: Minimum transit time to include (default 0.1 hours = 6 minutes)
    
    Returns:
        dict with mape, median_ape, and mape_samples keys
    """
    result = {
        'mape': None,
        'median_ape': None,
        'mape_samples': 0
    }
    
    try:
        # Filter out samples where actual transit time is too small
        # (to avoid exploding percentages when dividing by near-zero)
        mask = y_true_hours > min_threshold
        
        if mask.sum() == 0:
            # No samples with significant transit time
            return result
        
        y_true_filtered = y_true_hours[mask]
        y_pred_filtered = y_pred_hours[mask]
        
        # Calculate absolute percentage errors
        ape_values = np.abs((y_pred_filtered - y_true_filtered) / y_true_filtered) * 100
        
        # Filter out extreme outliers (APE > 1000%)
        # This can happen if predictions are way off
        ape_values = ape_values[ape_values < 1000]
        
        if len(ape_values) == 0:
            return result
        
        # Calculate MAPE and median APE
        result['mape'] = float(np.mean(ape_values))
        result['median_ape'] = float(np.median(ape_values))
        result['mape_samples'] = int(len(ape_values))
        
        # Sanity check - if MAPE is still extreme, set to None
        if result['mape'] > 500:
            result['mape'] = None
            result['median_ape'] = None
    
    except Exception as e:
        print(f"Warning: Error calculating MAPE: {e}")
    
    return result


def compute_selection_score(metrics, mae_weight=0.7, mape_weight=0.3):
    """
    Compute composite score for model selection.
    
    Args:
        metrics: dict with 'mae_hours' and optionally 'mape'
        mae_weight: weight for MAE contribution (default 0.7)
        mape_weight: weight for MAPE contribution (default 0.3)
    
    Returns:
        Lower score = better model
    """
    mae = metrics.get('mae_hours', float('inf'))
    
    # Validate MAE
    if mae is None or np.isnan(mae) or np.isinf(mae):
        mae = float('inf')
    
    # Use MAPE only if available and valid
    if (mape_weight > 0 and 
        metrics.get('mape') is not None and 
        not np.isnan(metrics['mape']) and 
        not np.isinf(metrics['mape']) and
        metrics.get('mape_samples', 0) > 10):  # Require at least 10 samples
        mape = metrics['mape']
        # Normalize MAPE to similar scale as MAE
        # Typical MAPE: 0-200%, Typical MAE: 0-10 hours
        # Scale MAPE by 0.05 to make 100% MAPE ≈ 5 hours equivalent
        normalized_mape = mape * 0.05
        score = mae * mae_weight + normalized_mape * mape_weight
    else:
        # Use MAE only (normalize weight to 1.0)
        score = mae
    
    return score


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics with safe None handling
    
    Args:
        metrics: Dictionary of metrics from compute_metrics
        prefix: Optional prefix for print statements (e.g., "Train: ", "Val: ")
    """
    print(f"\n{prefix}Metrics:")
    print("="*60)
    
    # Primary metrics
    if 'mae_hours' in metrics and metrics['mae_hours'] is not None:
        print(f"MAE (transit time):     {metrics['mae_hours']:.2f} hours")
    
    if 'median_error' in metrics and metrics['median_error'] is not None:
        print(f"Median AE:              {metrics['median_error']:.2f} hours")
    
    if 'rmse_hours' in metrics and metrics['rmse_hours'] is not None:
        print(f"RMSE:                   {metrics['rmse_hours']:.2f} hours")
    
    # Percentiles
    if all(k in metrics for k in ['p25_error', 'p75_error', 'p90_error']):
        if all(metrics[k] is not None for k in ['p25_error', 'p75_error', 'p90_error']):
            print(f"\nError Distribution:")
            print(f"  P25:  {metrics['p25_error']:.2f} hours")
            print(f"  P50:  {metrics['median_error']:.2f} hours")
            print(f"  P75:  {metrics['p75_error']:.2f} hours")
            print(f"  P90:  {metrics['p90_error']:.2f} hours")
            if 'p95_error' in metrics and metrics['p95_error'] is not None:
                print(f"  P95:  {metrics['p95_error']:.2f} hours")
            if 'p99_error' in metrics and metrics['p99_error'] is not None:
                print(f"  P99:  {metrics['p99_error']:.2f} hours")
    
    # MAPE if available
    if metrics.get('mape') is not None:
        print(f"\nMAPE (transit time):    {metrics['mape']:.2f}%")
        if metrics.get('median_ape') is not None:
            print(f"Median APE:             {metrics['median_ape']:.2f}%")
        print(f"MAPE samples:           {metrics.get('mape_samples', 0)}")
        print(f"  (calculated on transit times > 6 min)")
    else:
        print(f"\nMAPE:                   N/A (all transit times < 6 min)")
    
    # R² and bias
    if 'r2' in metrics and metrics['r2'] is not None:
        print(f"\nR² Score:               {metrics['r2']:.4f}")
    
    if 'mean_signed_error_hours' in metrics and metrics['mean_signed_error_hours'] is not None:
        bias = metrics['mean_signed_error_hours']
        if bias > 0:
            bias_dir = "over-estimating transit time"
        elif bias < 0:
            bias_dir = "under-estimating transit time"
        else:
            bias_dir = "neutral"
        print(f"Bias:                   {bias:+.2f} hours ({bias_dir})")
    
    # Directional errors
    if ('mae_hours_positive' in metrics and 'mae_hours_negative' in metrics and
        metrics['mae_hours_positive'] is not None and metrics['mae_hours_negative'] is not None):
        print(f"\nDirectional Errors:")
        print(f"  Over-estimation:  {metrics['mae_hours_positive']:.2f} hours "
              f"({metrics.get('n_positive_errors', 0)} samples)")
        print(f"  Under-estimation: {metrics['mae_hours_negative']:.2f} hours "
              f"({metrics.get('n_negative_errors', 0)} samples)")
    
    print("="*60)


class EarlyStopping:
    """Early stopping with best model tracking"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for metrics like R² (higher is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, metric, epoch=None):
        """
        Check if should stop training
        
        Args:
            metric: Current metric value
            epoch: Current epoch number (optional, for logging)
        
        Returns:
            True if should stop, False otherwise
        """
        # Validate metric
        if metric is None or np.isnan(metric) or np.isinf(metric):
            if self.verbose:
                print(f"  Warning: Invalid metric value: {metric}")
            return False
        
        score = float(metric)
        
        if self.best_score is None:
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
            if self.verbose:
                print(f"  Initial best score: {score:.4f}")
        else:
            # Check improvement based on mode
            if self.mode == 'min':
                improved = score < (self.best_score - self.min_delta)
            else:  # mode == 'max'
                improved = score > (self.best_score + self.min_delta)
            
            if improved:
                improvement = abs(score - self.best_score)
                self.best_score = score
                self.counter = 0
                if epoch is not None:
                    self.best_epoch = epoch
                if self.verbose:
                    print(f"  ✓ New best score: {score:.4f} (improved by {improvement:.4f})")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"  No improvement for {self.counter}/{self.patience} epochs "
                          f"(best: {self.best_score:.4f})")
                
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"\n  Early stopping triggered! Best epoch: {self.best_epoch}")
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


def compute_classification_metrics(y_pred, y_true, threshold=1.0):
    """
    Compute classification metrics for transit time prediction.
    Classify predictions as "fast" vs "slow" based on threshold.
    
    Args:
        y_pred: Predicted transit times in hours
        y_true: True transit times in hours
        threshold: Threshold in hours to classify as slow (default 1.0)
    
    Returns:
        Dictionary with classification metrics
    """
    # Flatten arrays
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    
    if len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'threshold_hours': float(threshold)
        }
    
    # Classify based on threshold
    pred_slow = y_pred > threshold
    true_slow = y_true > threshold
    
    # Confusion matrix components
    tp = np.sum(pred_slow & true_slow)
    tn = np.sum(~pred_slow & ~true_slow)
    fp = np.sum(pred_slow & ~true_slow)
    fn = np.sum(~pred_slow & true_slow)
    
    # Metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'threshold_hours': float(threshold)
    }


def compare_models(metrics_dict):
    """
    Compare multiple models and rank them
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
                     e.g., {'model_v1': metrics1, 'model_v2': metrics2}
    
    Returns:
        Sorted list of (model_name, score, metrics) tuples, best first
    """
    results = []
    
    for model_name, metrics in metrics_dict.items():
        score = compute_selection_score(metrics)
        results.append((model_name, score, metrics))
    
    # Sort by score (lower is better)
    results.sort(key=lambda x: x[1])
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for rank, (name, score, metrics) in enumerate(results, 1):
        mae = metrics.get('mae_hours')
        mape = metrics.get('mape')
        
        print(f"\n{rank}. {name}")
        print(f"   Score: {score:.2f}")
        
        if mae is not None:
            print(f"   MAE:   {mae:.2f} hours")
        else:
            print(f"   MAE:   N/A")
        
        if mape is not None:
            print(f"   MAPE:  {mape:.2f}%")
        else:
            print(f"   MAPE:  N/A")
    
    print("="*80 + "\n")
    
    return results