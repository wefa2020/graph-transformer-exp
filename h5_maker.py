#!/usr/bin/env python3
"""
prepare_data.py - Prepare H5 cache files directly to S3

Config is loaded from: s3://graph-transformer-exp/configs/config.json
Data paths are specified in config.
Dataset writes H5 files directly to S3.

Supports loading multiple JSON files from a local folder.
Only loads JSON files starting with "data_part" prefix.

python h5_maker.py --data-folder ./data/graph-data --config ./config_file/config.json
python h5_maker.py --data-folder s3://graph-transformer-exp/data/test.json --config s3://graph-transformer-exp/configs/config.json
"""

import os
import sys
import gc
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable

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

# Local data folder
LOCAL_DATA_FOLDER = "./data/graph-data"

# Package ID prefix filter
PACKAGE_ID_PREFIX = "TBA"

# JSON file prefix filter
JSON_FILE_PREFIX = "data_part"


class ProgressTracker:
    """Track and write progress to a file in the data folder."""
    
    def __init__(self, data_folder: str, total_records: int = 0):
        self.data_folder = Path(data_folder)
        self.progress_file = self.data_folder / "progress.json"
        self.total_records = total_records
        self.processed_records = 0
        self.current_stage = "initializing"
        self.start_time = datetime.now()
        self.stages = {
            "loading": 0,
            "filtering": 15,
            "preprocessing": 20,
            "train_h5": 30,
            "val_h5": 70,
            "test_h5": 85,
            "complete": 100
        }
        
        # Ensure data folder exists
        self.data_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress file
        self._write_progress()
    
    def _calculate_percentage(self) -> float:
        """Calculate overall completion percentage."""
        base_pct = self.stages.get(self.current_stage, 0)
        
        # Calculate progress within current stage
        if self.current_stage in ["train_h5", "val_h5", "test_h5"] and self.total_records > 0:
            stage_start = self.stages[self.current_stage]
            if self.current_stage == "train_h5":
                stage_end = self.stages["val_h5"]
            elif self.current_stage == "val_h5":
                stage_end = self.stages["test_h5"]
            else:
                stage_end = self.stages["complete"]
            
            stage_range = stage_end - stage_start
            progress_in_stage = (self.processed_records / self.total_records) * stage_range
            return min(stage_start + progress_in_stage, stage_end)
        
        return base_pct
    
    def _write_progress(self):
        """Write current progress to file."""
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "current_stage": self.current_stage,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "completion_percentage": round(self._calculate_percentage(), 2),
            "status": "running" if self.current_stage != "complete" else "complete"
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write progress file: {e}")
    
    def set_total_records(self, total: int):
        """Set total number of records."""
        self.total_records = total
        self._write_progress()
    
    def set_stage(self, stage: str, stage_total: int = 0):
        """Set current processing stage."""
        self.current_stage = stage
        self.processed_records = 0
        self.total_records = stage_total
        self._write_progress()
        logger.info(f"Progress: Stage '{stage}' started")
    
    def update(self, processed: int):
        """Update number of processed records."""
        self.processed_records = processed
        self._write_progress()
    
    def increment(self, count: int = 1):
        """Increment processed record count."""
        self.processed_records += count
        self._write_progress()
    
    def complete(self):
        """Mark processing as complete."""
        self.current_stage = "complete"
        self._write_progress()
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Progress: Complete! Total time: {elapsed:.1f}s")
    
    def get_progress_callback(self) -> Callable[[int, int], None]:
        """Return a callback function for dataset processing."""
        def callback(processed: int, total: int):
            self.processed_records = processed
            self.total_records = total
            self._write_progress()
        return callback


def filter_by_package_id_prefix(df: pd.DataFrame, prefix: str = PACKAGE_ID_PREFIX) -> pd.DataFrame:
    """
    Filter DataFrame to keep only records where package_id starts with specified prefix.
    
    Args:
        df: Input DataFrame
        prefix: Package ID prefix to filter by (default: 'TBA')
        
    Returns:
        Filtered DataFrame
    """
    if 'package_id' not in df.columns:
        logger.warning("Column 'package_id' not found in DataFrame. Skipping filter.")
        return df
    
    original_count = len(df)
    
    # Filter rows where package_id starts with prefix
    mask = df['package_id'].astype(str).str.startswith(prefix)
    filtered_df = df[mask].reset_index(drop=True)
    
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    logger.info(f"  Package ID filter (prefix='{prefix}'):")
    logger.info(f"    Original records:  {original_count:,}")
    logger.info(f"    Filtered records:  {filtered_count:,}")
    logger.info(f"    Removed records:   {removed_count:,} ({removed_count/original_count*100:.1f}%)")
    
    return filtered_df


def load_json_files_from_folder(
    folder_path: str, 
    file_prefix: str = JSON_FILE_PREFIX,
    progress: Optional[ProgressTracker] = None
) -> pd.DataFrame:
    """
    Load JSON files from a folder that start with specified prefix and concatenate them.
    
    Args:
        folder_path: Path to folder containing JSON files
        file_prefix: Only load files starting with this prefix (default: 'data_part')
        progress: Optional progress tracker
        
    Returns:
        Concatenated DataFrame from all matching JSON files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all JSON files
    all_json_files = sorted(folder.glob("*.json"))
    
    # Filter to only files starting with the specified prefix
    if file_prefix:
        json_files = [f for f in all_json_files if f.name.startswith(file_prefix)]
        skipped_files = [f for f in all_json_files if not f.name.startswith(file_prefix)]
        
        logger.info(f"JSON file filter (prefix='{file_prefix}'):")
        logger.info(f"  Total JSON files found: {len(all_json_files)}")
        logger.info(f"  Matching files:         {len(json_files)}")
        logger.info(f"  Skipped files:          {len(skipped_files)}")
        
        if skipped_files:
            logger.info(f"  Skipped file names:")
            for f in skipped_files[:10]:  # Show first 10 skipped files
                logger.info(f"    - {f.name}")
            if len(skipped_files) > 10:
                logger.info(f"    ... and {len(skipped_files) - 10} more")
    else:
        json_files = all_json_files
        logger.info(f"Found {len(json_files)} JSON file(s) (no prefix filter)")
    
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files starting with '{file_prefix}' found in: {folder_path}"
        )
    
    logger.info(f"\nLoading {len(json_files)} JSON file(s) from {folder_path}:")
    for f in json_files:
        logger.info(f"  - {f.name}")
    
    # Load and concatenate
    dfs = []
    total_records = 0
    
    for i, json_file in enumerate(json_files):
        logger.info(f"  Loading ({i+1}/{len(json_files)}): {json_file.name}")
        
        try:
            df = pd.read_json(json_file)
            record_count = len(df)
            total_records += record_count
            logger.info(f"    → {record_count:,} records")
            dfs.append(df)
        except Exception as e:
            logger.error(f"    → Failed to load: {e}")
            continue
        
        if progress:
            progress.update(i + 1)
    
    if not dfs:
        raise ValueError(f"No valid JSON files could be loaded from: {folder_path}")
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"\n  Combined: {len(combined_df):,} total records from {len(dfs)} files")
    
    return combined_df


def load_data(
    source_path: str, 
    file_prefix: str = JSON_FILE_PREFIX,
    progress: Optional[ProgressTracker] = None
) -> pd.DataFrame:
    """
    Load data from various sources:
    - S3 path (s3://...)
    - Local folder (loads all JSON files starting with prefix)
    - Local file (single JSON file)
    
    Args:
        source_path: Path to data source
        file_prefix: For folders, only load files starting with this prefix
        progress: Optional progress tracker
        
    Returns:
        DataFrame with loaded data
    """
    # S3 path
    if source_path.startswith('s3://'):
        import boto3
        import io
        logger.info(f"Loading from S3: {source_path}")
        path = source_path.replace('s3://', '')
        bucket, key = path.split('/', 1)
        response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        return pd.read_json(io.BytesIO(response['Body'].read()))
    
    # Local path
    path = Path(source_path)
    
    # If it's a directory, load all matching JSON files
    if path.is_dir():
        return load_json_files_from_folder(source_path, file_prefix=file_prefix, progress=progress)
    
    # If it's a single file
    if path.is_file():
        logger.info(f"Loading single file: {source_path}")
        return pd.read_json(source_path)
    
    raise FileNotFoundError(f"Path not found: {source_path}")


def load_distance_data(distance_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load distance matrix from S3 or local path."""
    if not distance_path:
        return None
    
    logger.info(f"  Loading distance matrix: {distance_path}")
    
    if distance_path.startswith('s3://'):
        import boto3
        import io
        path = distance_path.replace('s3://', '')
        bucket, key = path.split('/', 1)
        response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(response['Body'].read()))
    else:
        return pd.read_csv(distance_path)


def prepare_data(
    config: Config, 
    data_folder: str = None, 
    package_id_prefix: str = PACKAGE_ID_PREFIX,
    json_file_prefix: str = JSON_FILE_PREFIX
):
    """Load data and create H5 cache files directly to S3."""
    from data.data_preprocessor import PackageLifecyclePreprocessor
    from data.dataset import PackageLifecycleDataset
    
    # Determine data source
    data_source = data_folder if data_folder else config.data.source_data
    
    # Initialize progress tracker
    progress = ProgressTracker(data_source if data_folder else LOCAL_DATA_FOLDER)
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION")
    logger.info("=" * 70)
    logger.info(f"  Data source:       {data_source}")
    logger.info(f"  JSON file prefix:  {json_file_prefix if json_file_prefix else 'ALL'}")
    logger.info(f"  Package ID prefix: {package_id_prefix if package_id_prefix else 'ALL'}")
    logger.info(f"  Distance file:     {config.data.distance_file}")
    logger.info(f"  Cache dir:         {config.data.cache_dir}")
    logger.info(f"  Progress file:     {progress.progress_file}")
    logger.info("=" * 70)
    
    # Load data (handles folder, single file, or S3)
    progress.set_stage("loading")
    logger.info(f"\nLoading data from: {data_source}")
    df = load_data(data_source, file_prefix=json_file_prefix, progress=progress)
    logger.info(f"  Total samples loaded: {len(df):,}")
    
    # Filter by package_id prefix
    if package_id_prefix:
        progress.set_stage("filtering")
        logger.info(f"\nFiltering data by package_id prefix...")
        df = filter_by_package_id_prefix(df, prefix=package_id_prefix)
        
        if len(df) == 0:
            raise ValueError(f"No records remaining after filtering by package_id prefix '{package_id_prefix}'")
    else:
        logger.info("\nSkipping package_id prefix filter (disabled)")
    
    # Load distance matrix
    distance_df = load_distance_data(config.data.distance_file)
    
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
    
    # Fit preprocessor
    progress.set_stage("preprocessing")
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
    progress.set_stage("train_h5", len(train_df))
    logger.info("\n" + "=" * 70)
    logger.info("CREATING train.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=train_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.train_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info,
        progress_callback=progress.get_progress_callback()
    )
    logger.info(f"  Saved: {config.data.train_h5}")
    
    del train_df
    gc.collect()
    
    # Create val H5 directly to S3
    progress.set_stage("val_h5", len(val_df))
    logger.info("\n" + "=" * 70)
    logger.info("CREATING val.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=val_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.val_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info,
        progress_callback=progress.get_progress_callback()
    )
    logger.info(f"  Saved: {config.data.val_h5}")
    
    del val_df
    gc.collect()
    
    # Create test H5 directly to S3
    progress.set_stage("test_h5", len(test_df))
    logger.info("\n" + "=" * 70)
    logger.info("CREATING test.h5")
    logger.info("=" * 70)
    
    PackageLifecycleDataset(
        df=test_df,
        preprocessor=preprocessor,
        h5_cache_path=config.data.test_h5,
        load_from_cache=False,
        save_to_cache=True,
        log_fn=logger.info,
        progress_callback=progress.get_progress_callback()
    )
    logger.info(f"  Saved: {config.data.test_h5}")
    
    del test_df
    gc.collect()
    
    # Mark complete
    progress.complete()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  {config.data.train_h5}")
    logger.info(f"  {config.data.val_h5}")
    logger.info(f"  {config.data.test_h5}")
    logger.info(f"  {config.data.preprocessor_path}")
    logger.info(f"  Progress log: {progress.progress_file}")
    logger.info("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare H5 cache files')
    parser.add_argument(
        '--data-folder', 
        type=str, 
        default=LOCAL_DATA_FOLDER,
        help=f'Path to folder containing JSON files (default: {LOCAL_DATA_FOLDER})'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=CONFIG_PATH,
        help=f'Path to config file (default: {CONFIG_PATH})'
    )
    parser.add_argument(
        '--package-prefix',
        type=str,
        default=PACKAGE_ID_PREFIX,
        help=f'Package ID prefix to filter by (default: {PACKAGE_ID_PREFIX})'
    )
    parser.add_argument(
        '--file-prefix',
        type=str,
        default=JSON_FILE_PREFIX,
        help=f'JSON file name prefix to filter by (default: {JSON_FILE_PREFIX})'
    )
    parser.add_argument(
        '--no-package-filter',
        action='store_true',
        help='Disable package_id prefix filtering'
    )
    parser.add_argument(
        '--no-file-filter',
        action='store_true',
        help='Disable JSON file prefix filtering (load all .json files)'
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("H5 CACHE PREPARATION")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Data folder:     {args.data_folder}")
    logger.info(f"File prefix:     {args.file_prefix if not args.no_file_filter else 'DISABLED (all .json)'}")
    logger.info(f"Package prefix:  {args.package_prefix if not args.no_package_filter else 'DISABLED'}")
    
    config = Config.load(args.config)
    
    # Determine prefixes (None if filtering disabled)
    package_prefix = None if args.no_package_filter else args.package_prefix
    file_prefix = None if args.no_file_filter else args.file_prefix
    
    try:
        prepare_data(
            config, 
            data_folder=args.data_folder, 
            package_id_prefix=package_prefix,
            json_file_prefix=file_prefix
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()