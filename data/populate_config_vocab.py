#!/usr/bin/env python3
"""
scripts/populate_config_vocab.py - Populate config vocabulary from training data
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Set, List, Optional
from pathlib import Path


def extract_vocab_from_dataframe(
    df: pd.DataFrame,
    distance_df: Optional[pd.DataFrame] = None
) -> Dict[str, List[str]]:
    """
    Extract all vocabulary from training DataFrame.
    
    Args:
        df: Training DataFrame with 'events' column
        distance_df: Optional distance DataFrame with region info
    
    Returns:
        Dictionary with all vocabulary lists
    
    Raises:
        ValueError: If any required vocabulary is empty
    """
    locations: Set[str] = set()
    carriers: Set[str] = set()
    leg_types: Set[str] = set()
    ship_methods: Set[str] = set()
    regions: Set[str] = set()
    
    print(f"Extracting vocabulary from {len(df):,} samples...")
    
    if 'events' not in df.columns:
        raise ValueError("DataFrame must contain 'events' column")
    
    for idx, row in df.iterrows():
        events = row.get('events', [])
        if not isinstance(events, list):
            continue
        
        for event in events:
            event_type = str(event.get('event_type', ''))
            
            # Location (sort_center or delivery_station)
            if event_type == 'DELIVERY':
                station = event.get('delivery_station')
                if station and str(station) != 'nan':
                    locations.add(str(station))
            else:
                sort_center = event.get('sort_center')
                if sort_center and str(sort_center) != 'nan':
                    locations.add(str(sort_center))
            
            # Carrier
            carrier_id = event.get('carrier_id')
            if carrier_id and str(carrier_id) != 'nan':
                carriers.add(str(carrier_id))
            
            # Leg type
            leg_type = event.get('leg_type')
            if leg_type and str(leg_type) != 'nan':
                leg_types.add(str(leg_type))
            
            # Ship method
            ship_method = event.get('ship_method')
            if ship_method and str(ship_method) != 'nan':
                ship_methods.add(str(ship_method))
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1:,} samples...")
    
    # Extract regions from distance file
    if distance_df is not None:
        print("Extracting regions from distance data...")
        for col in ['super_region_1', 'super_region_2']:
            if col in distance_df.columns:
                for val in distance_df[col].dropna().unique():
                    if str(val).strip():
                        regions.add(str(val).strip())
    
    # Convert to sorted lists
    vocab = {
        'locations': sorted(list(locations)),
        'carriers': sorted(list(carriers)),
        'leg_types': sorted(list(leg_types)),
        'ship_methods': sorted(list(ship_methods)),
        'regions': sorted(list(regions)),
    }
    
    # Validate - error out if any vocabulary is empty
    errors = []
    for name, values in vocab.items():
        if not values:
            errors.append(f"  - {name}: no values found")
    
    if errors:
        raise ValueError(
            f"Failed to extract required vocabulary:\n" + "\n".join(errors) +
            "\n\nCheck that your data contains the expected fields."
        )
    
    print("\n=== Vocabulary Extracted ===")
    for name, values in vocab.items():
        print(f"  {name}: {len(values):,}")
    
    return vocab


def update_config_with_vocab(
    config_path: str,
    vocab: Dict[str, List[str]],
    output_path: Optional[str] = None,
):
    """
    Update config file with extracted vocabulary.
    
    Assumes event_types, problem_types, and zip_codes are already in config.
    
    Args:
        config_path: Path to existing config (local or S3)
        vocab: Vocabulary dictionary from extract_vocab_from_dataframe
        output_path: Output path (defaults to config_path)
    
    Raises:
        ValueError: If required vocab fields are missing from config
    """
    from config import Config
    
    print(f"\nLoading config from: {config_path}")
    config = Config.load(config_path)
    
    # Validate required fields exist in config
    required_fields = ['event_types', 'problem_types', 'zip_codes']
    missing = []
    for field in required_fields:
        value = getattr(config.data.vocab, field, None)
        if not value:
            missing.append(field)
    
    if missing:
        raise ValueError(
            f"Config missing required vocab fields: {missing}\n"
            "These must be configured manually in the config file."
        )
    
    # Update only the extracted vocabulary
    config.data.vocab.locations = vocab['locations']
    config.data.vocab.carriers = vocab['carriers']
    config.data.vocab.leg_types = vocab['leg_types']
    config.data.vocab.ship_methods = vocab['ship_methods']
    config.data.vocab.regions = vocab['regions']
    
    # Save
    save_path = output_path or config_path
    print(f"Saving config to: {save_path}")
    config.save(save_path)
    
    # Print summary
    print("\n=== Updated Config Vocabulary ===")
    sizes = config.get_vocab_sizes()
    for name, size in sizes.items():
        print(f"  {name}: {size} (including PAD, UNKNOWN)")
    
    return config


def main():
    """Main function to populate config from data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate config vocabulary from training data")
    parser.add_argument("--data", required=True, help="Path to training data (CSV, JSON, or Parquet)")
    parser.add_argument("--distance", help="Path to distance CSV file")
    parser.add_argument("--config", required=True, help="Path to existing config to update")
    parser.add_argument("--output", help="Output path for config (defaults to --config)")
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading data from: {args.data}")
    if args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    elif args.data.endswith('.json'):
        df = pd.read_json(args.data, lines=True)
    elif args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        raise ValueError(f"Unsupported file format: {args.data}")
    
    # Parse events column if it's a string
    if 'events' in df.columns and df['events'].dtype == object:
        if isinstance(df['events'].iloc[0], str):
            df['events'] = df['events'].apply(json.loads)
    
    print(f"Loaded {len(df):,} samples")
    
    # Load distance data
    distance_df = None
    if args.distance:
        print(f"Loading distance data from: {args.distance}")
        distance_df = pd.read_csv(args.distance)
    
    # Extract vocabulary
    vocab = extract_vocab_from_dataframe(df, distance_df)
    
    # Update config
    update_config_with_vocab(args.config, vocab, args.output)


if __name__ == "__main__":
    main()