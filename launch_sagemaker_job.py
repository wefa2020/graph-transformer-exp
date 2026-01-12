#!/usr/bin/env python3
"""
launch_sagemaker_job.py - Launch distributed training on SageMaker
PyTorch 2.5.1 / Python 3.11
"""

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import os


def launch_distributed_training(
    instance_type='ml.p3.16xlarge',
    instance_count=1,
    source_dir='.',
    entry_point='sagemaker_train.py',
    s3_data_path='s3://graph-transformer-exp/cache/',
    s3_output_path='s3://graph-transformer-exp/output/',
    role=None,
    max_run_hours=240,
    hyperparameters=None,
    use_spot_instances=False,
    max_wait_hours=48,
    volume_size=100,
):
    """Launch distributed training job on SageMaker"""
    
    session = sagemaker.Session()
    
    if role is None:
        role = sagemaker.get_execution_role()
    
    if hyperparameters is None:
        hyperparameters = {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'seed': 42,
            'patience': 15,
            'hidden_dim': 256,
            'num_layers': 40,
            'num_heads': 8,
            'dropout': 0.1,
        }
    
    source_dir = os.path.abspath(source_dir)
    entry_point_path = os.path.join(source_dir, entry_point)
    
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")
    if not os.path.exists(entry_point_path):
        raise ValueError(f"Entry point script does not exist: {entry_point_path}")
    
    # Always enable torch_distributed - works for single and multi-GPU
    distribution = {
        'torch_distributed': {
            'enabled': True
        }
    }
    
    # TensorBoard always enabled
    tensorboard_s3_path = f'{s3_output_path.rstrip("/")}/tensorboard'
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=tensorboard_s3_path,
        container_local_output_path='/opt/ml/output/tensorboard'
    )
    
    environment = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'eth0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    }
    
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': r'Train Loss: ([0-9\.]+)'},
        {'Name': 'train:mae', 'Regex': r'Train.*MAE: ([0-9\.]+)h'},
        {'Name': 'val:loss', 'Regex': r'Val Loss: ([0-9\.]+)'},
        {'Name': 'val:mae', 'Regex': r'Val.*MAE: ([0-9\.]+)h'},
        {'Name': 'val:r2', 'Regex': r'R²: ([0-9\.\-]+)'},
        {'Name': 'test:mae', 'Regex': r'Test.*MAE: ([0-9\.]+)h'},
        {'Name': 'test:r2', 'Regex': r'Test.*R²: ([0-9\.\-]+)'},
    ]
    
    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version='2.5.1',
        py_version='py311',
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        base_job_name='pkg-event-pred',
        max_run=max_run_hours * 3600,
        environment=environment,
        metric_definitions=metric_definitions,
        volume_size=volume_size,
        debugger_hook_config=False,
        disable_profiler=True,
        distribution=distribution,
        tensorboard_output_config=tensorboard_config,
        use_spot_instances=use_spot_instances,
        max_wait=max_wait_hours * 3600 if use_spot_instances else None,
    )
    
    inputs = {
        'training': sagemaker.inputs.TrainingInput(
            s3_data=s3_data_path,
            content_type='application/json',
        ),
    }
    
    print("=" * 80)
    print("SageMaker Distributed Training")
    print("=" * 80)
    print(f"PyTorch:        2.5.1")
    print(f"Python:         py311")
    print(f"Instance Type:  {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Volume Size:    {volume_size} GB")
    print(f"Max Run Time:   {max_run_hours} hours")
    print(f"Spot:           {use_spot_instances}")
    print(f"Source Dir:     {source_dir}")
    print(f"Entry Point:    {entry_point}")
    print(f"Data Path:      {s3_data_path}")
    print(f"Output Path:    {s3_output_path}")
    print(f"TensorBoard:    {tensorboard_s3_path}")
    print("-" * 80)
    print("Hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")
    print("=" * 80)
    
    estimator.fit(inputs, wait=True, logs='All')
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker distributed training')
    
    
    # Instance
    parser.add_argument('--instance-type', type=str, default='ml.p3.16xlarge')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--volume-size', type=int, default=100)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=40)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Source
    parser.add_argument('--source-dir', type=str, default='.')
    parser.add_argument('--entry-point', type=str, default='sagemaker_train.py')
    
    # Optional
    parser.add_argument('--role', type=str, default=None)
    parser.add_argument('--use-spot', action='store_true')
    parser.add_argument('--max-run-hours', type=int, default=240)
    parser.add_argument('--max-wait-hours', type=int, default=480)
    
    args = parser.parse_args()
    
    hyperparameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'seed': args.seed,
        'patience': 15,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
    }
    
    launch_distributed_training(
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role=args.role,
        hyperparameters=hyperparameters,
        source_dir=args.source_dir,
        entry_point=args.entry_point,
        use_spot_instances=args.use_spot,
        max_run_hours=args.max_run_hours,
        max_wait_hours=args.max_wait_hours,
        volume_size=args.volume_size,
    )