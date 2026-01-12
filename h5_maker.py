#!/usr/bin/env python3
"""
prepare_data.py - Prepare H5 cache files and upload to S3
"""

import os
import sys
import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare H5 cache files for training')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, 
                        default='/home/ubuntu/graph-transformer-exp/data/graph-data/12100106/',
                        help='Directory containing source data files')
    parser.add_argument('--data_file', type=str, default='package_lifecycles_batch_5.json',
                        help='Source JSON data file name')
    parser.add_argument('--distance_file', type=str, default='location_distances_complete.csv',
                        help='Distance matrix CSV file name')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Local directory for preprocessor and temp files')
    
    # Data splits
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    
    # S3 settings
    parser.add_argument('--s3_bucket', type=str, default='graph-transformer-exp',
                        help='S3 bucket name')
    parser.add_argument('--s3_prefix', type=str, default='package-lifecycle/cache/test',
                        help='S3 prefix/path for cache files')
    
    # Processing options
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    
    return parser.parse_args()


# =============================================================================
# S3 UTILITIES
# =============================================================================

def s3_upload_file(local_path: str, bucket: str, key: str):
    """Upload file to S3 with progress."""
    import boto3
    from boto3.s3.transfer import TransferConfig
    
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info(f"  Uploading {local_path} ({file_size_mb:.1f} MB) -> s3://{bucket}/{key}")
    
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,
    )
    
    boto3.client('s3').upload_file(local_path, bucket, key, Config=config)
    logger.info(f"  ✓ Uploaded to s3://{bucket}/{key}")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(args):
    """Load data and create H5 cache files, upload to S3."""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    from config import Config
    
    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # S3 paths
    s3_prefix = args.s3_prefix.strip('/')
    train_s3 = f"s3://{args.s3_bucket}/{s3_prefix}/train.h5"
    val_s3 = f"s3://{args.s3_bucket}/{s3_prefix}/val.h5"
    test_s3 = f"s3://{args.s3_bucket}/{s3_prefix}/test.h5"
    preprocessor_s3_key = f"{s3_prefix}/preprocessor.pkl"
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"  Data directory:     {data_dir}")
    logger.info(f"  Data file:          {args.data_file}")
    logger.info(f"  Distance file:      {args.distance_file}")
    logger.info(f"  Output directory:   {output_dir}")
    logger.info(f"  Train ratio:        {args.train_ratio}")
    logger.info(f"  Val ratio:          {args.val_ratio}")
    logger.info(f"  Test ratio:         {1 - args.train_ratio - args.val_ratio:.2f}")
    logger.info(f"  Random seed:        {args.seed}")
    logger.info(f"  S3 bucket:          {args.s3_bucket}")
    logger.info(f"  S3 prefix:          {s3_prefix}")
    logger.info("=" * 70)
    
    # Load raw data
    data_path = data_dir / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"\n  Loading data: {data_path}")
    df = pd.read_json(data_path)
    logger.info(f"  Total samples: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Load distance matrix
    distance_path = data_dir / args.distance_file
    distance_df = None
    if distance_path.exists():
        logger.info(f"  Loading distance matrix: {distance_path}")
        distance_df = pd.read_csv(distance_path)
        logger.info(f"  Distance matrix shape: {distance_df.shape}")
    else:
        logger.warning(f"  Distance file not found: {distance_path}")
        # Try parent directory
        alt_distance_path = data_dir.parent / args.distance_file
        if alt_distance_path.exists():
            logger.info(f"  Found in parent directory: {alt_distance_path}")
            distance_df = pd.read_csv(alt_distance_path)
    
    # Shuffle and split
    logger.info(f"\n  Shuffling with seed={args.seed}")
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    logger.info(f"\n  Data split:")
    logger.info(f"    - Train: {len(train_df):,} samples ({args.train_ratio*100:.0f}%)")
    logger.info(f"    - Val:   {len(val_df):,} samples ({args.val_ratio*100:.0f}%)")
    logger.info(f"    - Test:  {len(test_df):,} samples ({(1-args.train_ratio-args.val_ratio)*100:.0f}%)")
    
    del df
    gc.collect()
    
    # Create preprocessor
    logger.info("\n" + "=" * 70)
    logger.info("FITTING PREPROCESSOR")
    logger.info("=" * 70)
    
    config = Config()
    preprocessor = PackageLifecyclePreprocessor(config=config, distance_df=distance_df)
    preprocessor.fit(train_df)
    
    # Save preprocessor locally
    preprocessor_local = output_dir / 'preprocessor.pkl'
    preprocessor.save(str(preprocessor_local))
    logger.info(f"  ✓ Saved preprocessor: {preprocessor_local}")
    
    # Upload preprocessor to S3
    s3_upload_file(str(preprocessor_local), args.s3_bucket, preprocessor_s3_key)
    
    del distance_df
    gc.collect()
    
    # Create train H5 and upload to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING train.h5 AND UPLOADING TO S3")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=train_df,
        preprocessor=preprocessor,
        h5_cache_path=train_s3,  # S3 path directly
        load_from_cache=False,
        save_to_cache=True,
        num_workers=args.num_workers,
        log_fn=logger.info
    )
    
    del train_df
    gc.collect()
    
    # Create val H5 and upload to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING val.h5 AND UPLOADING TO S3")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=val_df,
        preprocessor=preprocessor,
        h5_cache_path=val_s3,  # S3 path directly
        load_from_cache=False,
        save_to_cache=True,
        num_workers=args.num_workers,
        log_fn=logger.info
    )
    
    del val_df
    gc.collect()
    
    # Create test H5 and upload to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING test.h5 AND UPLOADING TO S3")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=test_df,
        preprocessor=preprocessor,
        h5_cache_path=test_s3,  # S3 path directly
        load_from_cache=False,
        save_to_cache=True,
        num_workers=args.num_workers,
        log_fn=logger.info
    )
    
    del test_df
    gc.collect()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  S3 files created:")
    logger.info(f"    - {train_s3}")
    logger.info(f"    - {val_s3}")
    logger.info(f"    - {test_s3}")
    logger.info(f"    - s3://{args.s3_bucket}/{preprocessor_s3_key}")
    logger.info("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    logger.info("\n" + "=" * 70)
    logger.info("H5 CACHE PREPARATION SCRIPT")
    logger.info("=" * 70)
    
    try:
        prepare_data(args)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()