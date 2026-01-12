import csv
import json

def process_zip_codes(input_file='code.csv', output_file='output.json', records_per_line=20):
    """
    Read ZIP codes from CSV, output as a single JSON array (20 items per line)
    """
    
    # Step 1: Read the CSV and extract 'zip' column
    zip_codes = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            zip_code = str(row['zip']).strip()
            zip_codes.append(zip_code)
    
    print(f"Total ZIP codes read: {len(zip_codes)}")
    
    # Step 2: Format as valid JSON with 20 items per line
    lines = []
    for i in range(0, len(zip_codes), records_per_line):
        chunk = zip_codes[i:i + records_per_line]
        line = ','.join(json.dumps(z) for z in chunk)
        lines.append(line)
    
    # Step 3: Write to output file as valid JSON
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('[\n')
        file.write(',\n'.join(lines))
        file.write('\n]')
    
    print(f"Output written to: {output_file}")

# Run the script
if __name__ == '__main__':
    process_zip_codes()