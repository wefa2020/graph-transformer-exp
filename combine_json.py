
import json
import os
from pathlib import Path
from tqdm import tqdm

def combine_json_files(input_dir='/home/ubuntu/graph-transformer-exp/data/graph-data/12100106/', output_file='all.json'):
    """
    Combine all JSON files in a directory into one all.json file
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output filename (will be saved in input_dir)
    """
    
    input_path = Path(input_dir)
    output_path = input_path / output_file
    
    # Find all JSON files (excluding the output file if it exists)
    json_files = [
        f for f in input_path.glob('*.json') 
        if f.name != output_file
    ]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Combine all data
    combined_data = []
    total_records = 0
    
    print(f"\nCombining files...")
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                combined_data.extend(data)
                total_records += len(data)
            else:
                print(f"Warning: Unexpected data type in {json_file.name}: {type(data)}")
        
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Save combined data
    print(f"\nSaving {total_records} records to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    # Print statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*80}")
    print(f"COMBINATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total files processed: {len(json_files)}")
    print(f"Total records: {total_records}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*80}")

if __name__ == '__main__':
    combine_json_files()