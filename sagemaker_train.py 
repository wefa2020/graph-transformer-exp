# sagemaker_train.py - SageMaker entry point for distributed training

import subprocess
import sys
import os


def install_dependencies():
    """Install required packages"""
    packages = [
        'torch-geometric',
        'tensorboard',
        'tqdm',
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])


if __name__ == '__main__':
    # Install dependencies if needed
    # install_dependencies()
    
    # Import and run the main training script
    from train_distributed import main
    main()