#!/usr/bin/env python3
"""
prepare_data.py - Prepare H5 cache files directly to S3

Config is loaded from: s3://graph-transformer-exp/configs/config.json
Data paths are specified in config.
Dataset writes H5 files directly to S3.
"""

import os
import sys
import gc
import logging
from pathlib import Path

import pandas as pd

from config import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Fixed config location
CONFIG_PATH = "s3://graph-transformer-exp/configs/config.json"


def prepare_data(config: Config):
    """Load data and create H5 cache files directly to S3."""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION")
    logger.info("=" * 70)
    logger.info(f"  Data file:      {config.data.source_data}")
    logger.info(f"  Distance file:  {config.data.distance_file}")
    logger.info(f"  Cache dir:      {config.data.cache_dir}")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"\nLoading data: {config.data.source_data}")
    
    if config.data.source_data.startswith('s3://'):
        import boto3
        import io
        path = config.data.source_data.replace('s3://', '')
        bucket, key = path.split('/', 1)
        response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        df = pd.read_json(io.BytesIO(response['Body'].read()))
    else:
        df = pd.read_json(config.data.source_data)
    
    logger.info(f"  Total samples: {len(df):,}")
    
    # Load distance matrix
    distance_df = None
    if config.data.distance_file:
        logger.info(f"  Loading distance matrix: {config.data.distance_file}")
        if config.data.distance_file.startswith('s3://'):
            import boto3
            import io
            path = config.data.distance_file.replace('s3://', '')
            bucket, key = path.split('/', 1)
            response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            distance_df = pd.read_csv(io.BytesIO(response['Body'].read()))
        else:
            distance_df = pd.read_csv(config.data.distance_file)
    
    # Shuffle and split (90/5/5)
    seed = config.training.seed
    logger.info(f"\nShuffling with seed={seed}")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(0.9 * n_total)
    n_val = int(0.05 * n_total)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    logger.info(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    del df
    gc.collect()
    
    # Fit preprocessor - PASS CONFIG HERE
    logger.info("\n" + "=" * 70)
    logger.info("FITTING PREPROCESSOR")
    logger.info("=" * 70)
    
    preprocessor = PackageLifecyclePreprocessor(config=config, distance_df=distance_df)
    preprocessor.fit(train_df)
    
    # Save preprocessor directly to S3
    preprocessor.save(config.data.preprocessor_path)
    logger.info(f"  Saved: {config.data.preprocessor_path}")
    
    del distance_df
    gc.collect()
    
    # Create train H5 directly to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING train.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=train_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.train_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    logger.info(f"  Saved: {config.data.train_h5}")
    
    del train_df
    gc.collect()
    
    # Create val H5 directly to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING val.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=val_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.val_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    logger.info(f"  Saved: {config.data.val_h5}")
    
    del val_df
    gc.collect()
    
    # Create test H5 directly to S3
    logger.info("\n" + "=" * 70)
    logger.info("CREATING test.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=test_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.test_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info
    )
    logger.info(f"  Saved: {config.data.test_h5}")
    
    del test_df
    gc.collect()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  {config.data.train_h5}")
    logger.info(f"  {config.data.val_h5}")
    logger.info(f"  {config.data.test_h5}")
    logger.info(f"  {config.data.preprocessor_path}")
    logger.info("=" * 70)


def main():
    logger.info("=" * 70)
    logger.info("H5 CACHE PREPARATION")
    logger.info("=" * 70)
    logger.info(f"Loading config from: {CONFIG_PATH}")
    
    config = Config.load(CONFIG_PATH)
    
    try:
        prepare_data(config)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()