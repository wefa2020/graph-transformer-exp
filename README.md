## Step 1: Extract Data from Neptune

python extract_data.py --start-date "2025-01-01T00:00:00Z" --end-date "2025-01-31T23:59:59Z" --output package_lifecycles.json

## Step 2: Train Model
# Basic training
python train.py

# With custom config
python train.py --config configs/custom_config.yaml

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pt

## Step 3: Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt

## Step 4: Inference

# Single prediction
python inference.py --checkpoint checkpoints/best_model.pt --input sample_lifecycle.json

# Batch prediction
python inference.py --checkpoint checkpoints/best_model.pt --input batch_lifecycles.json --output predictions.json

## Monitoring
tensorboard --logdir logs/



