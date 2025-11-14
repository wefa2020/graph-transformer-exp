import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def plot_training_history(log_dir):
    """Plot training history from tensorboard logs"""
    from tensorboard.backend.event_processing import event_accumulator
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get metrics
    train_loss = [(s.step, s.value) for s in ea.Scalars('train/loss')]
    val_loss = [(s.step, s.value) for s in ea.Scalars('val/loss')]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot([x[0] for x in train_loss], [x[1] for x in train_loss], label='Train')
    axes[0].plot([x[0] for x in val_loss], [x[1] for x in val_loss], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE
    train_mae = [(s.step, s.value) for s in ea.Scalars('train/mae')]
    val_mae = [(s.step, s.value) for s in ea.Scalars('val/mae')]
    
    axes[1].plot([x[0] for x in train_mae], [x[1] for x in train_mae], label='Train')
    axes[1].plot([x[0] for x in val_mae], [x[1] for x in val_mae], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

def plot_predictions_vs_actual(predictions, actuals, preprocessor):
    """Plot predicted vs actual event times"""
    
    # Inverse transform
    predictions = preprocessor.inverse_transform_time(predictions.reshape(-1, 1)).flatten()
    actuals = preprocessor.inverse_transform_time(actuals.reshape(-1, 1)).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Time (hours)')
    axes[0].set_ylabel('Predicted Time (hours)')
    axes[0].set_title('Predicted vs Actual Event Times')
    axes[0].grid(True)
    
    # Error distribution
    errors = predictions - actuals
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Prediction Error (hours)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution (MAE: {np.abs(errors).mean():.2f}h)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300)
    plt.show()

def plot_lifecycle_prediction(lifecycle, predictions, preprocessor):
    """Visualize predictions for a single package lifecycle"""
    
    events = lifecycle['events']
    event_times = [datetime.fromisoformat(e['event_time'].replace('Z', '+00:00')) 
                   for e in events]
    
    # Convert to hours from start
    start_time = event_times[0]
    actual_hours = [(t - start_time).total_seconds() / 3600 for t in event_times]
    
    # Predicted cumulative times
    predicted_deltas = preprocessor.inverse_transform_time(predictions.reshape(-1, 1)).flatten()
    predicted_hours = [0]  # Start at 0
    for delta in predicted_deltas:
        predicted_hours.append(predicted_hours[-1] + delta)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    event_types = [e['event_type'] for e in events]
    
    ax.plot(range(len(actual_hours)), actual_hours, 'o-', label='Actual', markersize=8)
    ax.plot(range(len(predicted_hours)), predicted_hours, 's--', label='Predicted', markersize=8)
    
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Time from Start (hours)')
    ax.set_title(f'Package {lifecycle["package_id"]} - Event Timeline')
    ax.set_xticks(range(len(event_types)))
    ax.set_xticklabels(event_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'lifecycle_prediction_{lifecycle["package_id"]}.png', dpi=300)
    plt.show()