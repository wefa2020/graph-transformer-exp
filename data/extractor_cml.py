#!/usr/bin/env python3
"""
Script to extract package lifecycle data from Neptune using Gremlin
"""

import argparse
import sys
from data.neptune_extractor import NeptuneDataExtractor
from config import Config

def main():
    parser = argparse.ArgumentParser(
        description='Extract package lifecycle data from Neptune using Gremlin'
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        help='Neptune endpoint (default: from config)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in ISO format (e.g., 2025-01-01T00:00:00Z)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in ISO format (e.g., 2025-01-31T23:59:59Z)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/graph-data',
        help='Output directory for JSON files (default: data/graph-data)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for saving JSON files (default: 100)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of worker threads (default: 10)'
    )
    parser.add_argument(
        '--use-iam',
        action='store_true',
        help='Use IAM authentication'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    endpoint = args.endpoint or config.neptune.endpoint
    
    if not endpoint or endpoint == "your-neptune-endpoint:8182":
        print("ERROR: Neptune endpoint not configured!")
        print("Please set endpoint in config.py or use --endpoint argument")
        sys.exit(1)
    
    print("="*80)
    print("NEPTUNE DATA EXTRACTION")
    print("="*80)
    print(f"Endpoint: {endpoint}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Use IAM: {args.use_iam}")
    print("="*80)
    print()
    
    # Create extractor
    try:
        extractor = NeptuneDataExtractor(
            endpoint=endpoint,
            use_iam=args.use_iam,
            max_workers=args.max_workers
        )
        
        # Extract lifecycles
        df = extractor.extract_lifecycles(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
        print(f"\n✓ Extraction complete!")
        print(f"✓ Extracted {len(df)} valid package lifecycles")
        print(f"✓ Data saved to: {args.output_dir}")
        print(f"\nTo train the model, run:")
        print(f"  python train.py --data {args.output_dir}/package_lifecycles_complete.json")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            extractor.close()
        except:
            pass

if __name__ == '__main__':
    main()