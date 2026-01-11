#!/usr/bin/env python3
"""
prepare_data.py - Prepare H5 cache files locally and upload to S3
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
    parser.add_argument('--data_dir', type=str, default='/ubuntu/graph-transformer-exp/data/graph-data/12100106/',
                        help='Directory containing source data files')
    parser.add_argument('--data_file', type=str, default='source.json',
                        help='Source JSON data file name')
    parser.add_argument('--distance_file', type=str, default='location_distances_complete.csv',
                        help='Distance matrix CSV file name')
    
    # Output settings
    parser.add_argument('--output_subdir', type=str, default='cache',
                        help='Subdirectory within data_dir for H5 files')
    parser.add_argument('--preprocessor_dir', type=str, default='./output',
                        help='Directory to save preprocessor pickle')
    
    # Data splits
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    
    # S3 settings
    parser.add_argument('--s3_bucket', type=str, default="graph-transformer-exp")
    parser.add_argument('--s3_prefix', type=str, default='package-lifecycle/cache/',
                        help='S3 prefix/path for cache files')
    parser.add_argument('--upload_to_s3', action='store_true',
                        help='Upload H5 files to S3 after generation')
    parser.add_argument('--skip_if_exists', action='store_true',
                        help='Skip generation if local H5 files already exist')
    
    # Processing options
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files without asking')
    
    return parser.parse_args()


# =============================================================================
# S3 UTILITIES
# =============================================================================

def s3_exists(bucket: str, key: str) -> bool:
    """Check if S3 object exists."""
    import boto3
    from botocore.exceptions import ClientError
    
    try:
        boto3.client('s3').head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def s3_upload(local_path: str, bucket: str, key: str):
    """Upload file to S3."""
    import boto3
    
    file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
    logger.info(f"  Uploading {local_path} ({file_size:.1f} MB) -> s3://{bucket}/{key}")
    
    # Use multipart upload for large files
    from boto3.s3.transfer import TransferConfig
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100 MB
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,  # 100 MB
    )
    
    boto3.client('s3').upload_file(
        local_path, bucket, key,
        Config=config,
        Callback=ProgressCallback(local_path, file_size) if file_size > 100 else None
    )
    logger.info(f"  ✓ Uploaded to s3://{bucket}/{key}")


class ProgressCallback:
    """Callback for S3 upload progress."""
    def __init__(self, filename: str, size_mb: float):
        self.filename = filename
        self.size_mb = size_mb
        self.uploaded = 0
        self.last_percent = 0
    
    def __call__(self, bytes_transferred):
        self.uploaded += bytes_transferred
        percent = int((self.uploaded / (self.size_mb * 1024 * 1024)) * 100)
        if percent >= self.last_percent + 10:
            logger.info(f"    Upload progress: {percent}%")
            self.last_percent = percent


def upload_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str, file_patterns: list = None):
    """Upload all matching files from local directory to S3."""
    import glob
    
    if file_patterns is None:
        file_patterns = ['*.h5', '*.pkl']
    
    uploaded_files = []
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(local_dir, pattern))
        for local_file in files:
            filename = os.path.basename(local_file)
            s3_key = f"{s3_prefix.rstrip('/')}/{filename}"
            s3_upload(local_file, bucket, s3_key)
            uploaded_files.append(f"s3://{bucket}/{s3_key}")
    
    return uploaded_files


# =============================================================================
# DATA PREPARATION
# =============================================================================

def check_local_cache_exists(cache_dir: str) -> dict:
    """Check which cache files exist locally."""
    files = {
        'train': os.path.join(cache_dir, 'train.h5'),
        'val': os.path.join(cache_dir, 'val.h5'),
        'test': os.path.join(cache_dir, 'test.h5'),
        'preprocessor': os.path.join(cache_dir, 'preprocessor.pkl')
    }
    
    status = {}
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024 * 1024) if exists else 0
        status[name] = {'path': path, 'exists': exists, 'size_mb': size}
    
    return status


def prepare_data(args):
    """Load data and create H5 cache files."""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    from config import Config
    
    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    cache_dir = data_dir / args.output_subdir
    
    # Local file paths
    train_h5 = cache_dir / 'train.h5'
    val_h5 = cache_dir / 'val.h5'
    test_h5 = cache_dir / 'test.h5'
    preprocessor_pkl = cache_dir / 'preprocessor.pkl'
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"  Data directory:     {data_dir}")
    logger.info(f"  Data file:          {args.data_file}")
    logger.info(f"  Distance file:      {args.distance_file}")
    logger.info(f"  Cache directory:    {cache_dir}")
    logger.info(f"  Train ratio:        {args.train_ratio}")
    logger.info(f"  Val ratio:          {args.val_ratio}")
    logger.info(f"  Test ratio:         {1 - args.train_ratio - args.val_ratio:.2f}")
    logger.info(f"  Random seed:        {args.seed}")
    logger.info(f"  Upload to S3:       {args.upload_to_s3}")
    if args.upload_to_s3:
        logger.info(f"  S3 bucket:          {args.s3_bucket}")
        logger.info(f"  S3 prefix:          {args.s3_prefix}")
    logger.info("=" * 70)
    
    # Check if local cache exists
    cache_status = check_local_cache_exists(str(cache_dir))
    all_exist = all(v['exists'] for v in cache_status.values())
    
    if all_exist:
        logger.info("\n  Local cache files already exist:")
        for name, info in cache_status.items():
            logger.info(f"    - {name}: {info['path']} ({info['size_mb']:.1f} MB)")
        
        if args.skip_if_exists:
            logger.info("\n  Skipping generation (--skip_if_exists)")
            if args.upload_to_s3:
                logger.info("\n  Proceeding to S3 upload...")
                _upload_to_s3(args, cache_dir)
            return
        
        if not args.overwrite:
            response = input("\n  Do you want to overwrite? (y/N): ")
            if response.lower() != 'y':
                logger.info("  Skipping cache creation.")
                if args.upload_to_s3:
                    response2 = input("  Upload existing files to S3? (y/N): ")
                    if response2.lower() == 'y':
                        _upload_to_s3(args, cache_dir)
                return
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    data_path = data_dir / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"\n  Loading data: {data_path}")
    df = pd.read_json(data_path)
    logger.info(f"  Total samples: {len(df):,}")
    
    # Show sample columns
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
    n_test = n_total - n_train - n_val
    
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
    
    # Save preprocessor
    preprocessor.save(str(preprocessor_pkl))
    logger.info(f"  ✓ Saved preprocessor: {preprocessor_pkl}")
    
    # Also save to preprocessor_dir if different
    if args.preprocessor_dir:
        os.makedirs(args.preprocessor_dir, exist_ok=True)
        alt_preprocessor_path = os.path.join(args.preprocessor_dir, 'preprocessor.pkl')
        preprocessor.save(alt_preprocessor_path)
        logger.info(f"  ✓ Saved preprocessor copy: {alt_preprocessor_path}")
    
    del distance_df
    gc.collect()
    
    # Create train H5
    logger.info("\n" + "=" * 70)
    logger.info("CREATING train.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=train_df,
        preprocessor=preprocessor,
        h5_cache_path=str(train_h5),
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    
    train_size = train_h5.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Created: {train_h5} ({train_size:.1f} MB)")
    
    del train_df
    gc.collect()
    
    # Create val H5
    logger.info("\n" + "=" * 70)
    logger.info("CREATING val.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=val_df,
        preprocessor=preprocessor,
        h5_cache_path=str(val_h5),
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    
    val_size = val_h5.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Created: {val_h5} ({val_size:.1f} MB)")
    
    del val_df
    gc.collect()
    
    # Create test H5
    logger.info("\n" + "=" * 70)
    logger.info("CREATING test.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=test_df,
        preprocessor=preprocessor,
        h5_cache_path=str(test_h5),
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    
    test_size = test_h5.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Created: {test_h5} ({test_size:.1f} MB)")
    
    del test_df
    gc.collect()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("LOCAL CACHE CREATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Cache directory: {cache_dir}")
    logger.info(f"  Files created:")
    logger.info(f"    - train.h5:        {train_size:.1f} MB")
    logger.info(f"    - val.h5:          {val_size:.1f} MB")
    logger.info(f"    - test.h5:         {test_size:.1f} MB")
    logger.info(f"    - preprocessor.pkl")
    logger.info(f"  Total size: {train_size + val_size + test_size:.1f} MB")
    logger.info("=" * 70)
    
    # Upload to S3 if requested
    if args.upload_to_s3:
        _upload_to_s3(args, cache_dir)


def _upload_to_s3(args, cache_dir):
    """Upload cache files to S3."""
    if not args.s3_bucket:
        logger.error("  S3 bucket not specified! Use --s3_bucket")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("UPLOADING TO S3")
    logger.info("=" * 70)
    logger.info(f"  Bucket: {args.s3_bucket}")
    logger.info(f"  Prefix: {args.s3_prefix}")
    
    try:
        uploaded = upload_directory_to_s3(
            str(cache_dir),
            args.s3_bucket,
            args.s3_prefix,
            file_patterns=['*.h5', '*.pkl']
        )
        
        logger.info("\n  ✓ Upload complete!")
        logger.info(f"  Uploaded files:")
        for f in uploaded:
            logger.info(f"    - {f}")
        
    except Exception as e:
        logger.error(f"  S3 upload failed: {e}")
        import traceback
        traceback.print_exc()


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