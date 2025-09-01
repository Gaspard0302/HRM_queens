#!/usr/bin/env python3

import os
import re
import json

def extract_level_data(content):
    """Extract level data from TypeScript content."""
    # Extract the size
    size_match = re.search(r'size:\s*(\d+)', content)
    if not size_match:
        return None
    size = int(size_match.group(1))
    
    # Find the colorRegions section by looking for the pattern more carefully
    # Look for all rows that are arrays of strings
    rows_pattern = r'\[["A-Z, ]+\]'
    
    # Find the start of colorRegions
    start_match = re.search(r'colorRegions:\s*\[', content)
    if not start_match:
        return None
    
    # Get content starting from colorRegions
    from_regions = content[start_match.end()-1:]
    
    # Find all row arrays in the colorRegions section
    rows = re.findall(rows_pattern, from_regions)
    
    color_regions = []
    for row in rows[:size]:  # Take only the first 'size' rows
        # Extract the cells from each row
        cells = re.findall(r'"([A-Z])"', row)
        if cells and len(cells) == size:  # Ensure row has correct number of cells
            color_regions.append(cells)
    
    if len(color_regions) != size:
        return None
    
    return {
        "size": size,
        "colorRegions": color_regions
    }

def convert_file(input_path, output_path):
    """Convert a single TypeScript file to JSON."""
    with open(input_path, 'r') as f:
        content = f.read()
    
    level_data = extract_level_data(content)
    if level_data:
        with open(output_path, 'w') as f:
            json.dump(level_data, f, indent=2)
        return True
    return False

def main():
    input_dir = 'queens_levels'
    output_dir = 'queens_levels_json'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    converted = 0
    failed = 0
    size_8_count = 0
    other_sizes = {}
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.ts') and filename.startswith('level'):
            input_path = os.path.join(input_dir, filename)
            
            with open(input_path, 'r') as f:
                content = f.read()
            
            # Check the size
            size_match = re.search(r'size:\s*(\d+)', content)
            if size_match:
                size = int(size_match.group(1))
                if size == 8:
                    size_8_count += 1
                    # Only convert size 8 files
                    output_filename = filename.replace('.ts', '.json')
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if convert_file(input_path, output_path):
                        converted += 1
                        print(f"Converted: {filename} (size {size})")
                    else:
                        failed += 1
                        print(f"Failed to convert: {filename} (size {size})")
                else:
                    other_sizes[size] = other_sizes.get(size, 0) + 1
    
    print(f"\nConversion complete:")
    print(f"  Size 8 files: {size_8_count}")
    print(f"  Successfully converted: {converted}")
    print(f"  Failed to convert: {failed}")
    print(f"\nOther sizes found:")
    for size, count in sorted(other_sizes.items()):
        print(f"  Size {size}: {count} files")

if __name__ == "__main__":
    main()
