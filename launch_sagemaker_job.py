# launch_sagemaker_job.py - Launch distributed training on SageMaker

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import boto3
from datetime import datetime


def launch_distributed_training(
    instance_type='ml.p3.16xlarge',  # 8 GPUs
    instance_count=1,
    num_gpus_per_instance=8,
    source_dir='.',
    entry_point='sagemaker_train.py',
    s3_data_path='s3://your-bucket/data/',
    s3_output_path='s3://your-bucket/output/',
    role=None,
    max_run_hours=24,
    hyperparameters=None
):
    """
    Launch distributed training job on SageMaker
    
    Args:
        instance_type: EC2 instance type (e.g., 'ml.p3.16xlarge', 'ml.p4d.24xlarge')
        instance_count: Number of instances
        num_gpus_per_instance: Number of GPUs per instance
        source_dir: Directory containing training scripts
        entry_point: Entry point script
        s3_data_path: S3 path to training data
        s3_output_path: S3 path for output
        role: SageMaker execution role ARN
        max_run_hours: Maximum training time in hours
        hyperparameters: Training hyperparameters
    """
    
    # Get SageMaker session and role
    session = sagemaker.Session()
    
    if role is None:
        role = sagemaker.get_execution_role()
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'epochs': 100,
            'batch-size': 64,
            'learning-rate': 1e-4,
            'seed': 42,
        }
    
    # Calculate total GPUs
    total_gpus = instance_count * num_gpus_per_instance
    
    # TensorBoard configuration
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f'{s3_output_path}/tensorboard',
        container_local_output_path='/opt/ml/output/tensorboard'
    )
    
    # Job name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'package-event-prediction-{timestamp}'
    
    # Distribution configuration for SMDP
    distribution = {
        'smdistributed': {
            'dataparallel': {
                'enabled': True
            }
        }
    }
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version='2.0.1',  # or your PyTorch version
        py_version='py310',
        distribution=distribution,
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        base_job_name=job_name,
        max_run=max_run_hours * 3600,
        tensorboard_output_config=tensorboard_output_config,
        debugger_hook_config=False,  # Disable debugger for performance
        
        # Environment variables
        environment={
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'eth0',
        },
        
        # Metric definitions for CloudWatch
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Train - Loss: ([0-9\\.]+)'},
            {'Name': 'train:mae_hours', 'Regex': 'Train - Loss: [0-9\\.]+ MAE: ([0-9\\.]+)h'},
            {'Name': 'val:loss', 'Regex': 'Val   - Loss: ([0-9\\.]+)'},
            {'Name': 'val:mae_hours', 'Regex': 'Val   - Loss: [0-9\\.]+ MAE: ([0-9\\.]+)h'},
            {'Name': 'val:r2', 'Regex': 'R²: ([0-9\\.]+)'},
        ],
    )
    
    # Define input channels
    inputs = {
        'training': s3_data_path,
    }
    
    print(f"="*80)
    print(f"Launching SageMaker Distributed Training Job")
    print(f"="*80)
    print(f"Job name: {job_name}")
    print(f"Instance type: {instance_type}")
    print(f"Instance count: {instance_count}")
    print(f"GPUs per instance: {num_gpus_per_instance}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Data path: {s3_data_path}")
    print(f"Output path: {s3_output_path}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"="*80)
    
    # Start training
    estimator.fit(inputs, job_name=job_name, wait=True)
    
    return estimator


def launch_multi_node_training(
    instance_type='ml.p3.16xlarge',
    instance_count=2,  # Multi-node
    **kwargs
):
    """Launch multi-node distributed training"""
    return launch_distributed_training(
        instance_type=instance_type,
        instance_count=instance_count,
        **kwargs
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker distributed training')
    parser.add_argument('--instance-type', type=str, default='ml.p3.16xlarge')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--s3-data', type=str, required=True, help='S3 path to training data')
    parser.add_argument('--s3-output', type=str, required=True, help='S3 path for output')
    parser.add_argument('--role', type=str, default=None, help='SageMaker execution role')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    hyperparameters = {
        'epochs': args.epochs,
        'batch-size': args.batch_size,
        'learning-rate': args.learning_rate,
        'seed': 42,
    }
    
    estimator = launch_distributed_training(
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        s3_data_path=args.s3_data,
        s3_output_path=args.s3_output,
        role=args.role,
        hyperparameters=hyperparameters,
    )