#!/usr/bin/env python3

import re

with open('queens_levels/level1.ts', 'r') as f:
    content = f.read()

print("Content length:", len(content))
print("\nLooking for colorRegions...")

# Try different patterns
patterns = [
    r'colorRegions:\s*\[(.*?)\]',
    r'colorRegions:\s*\[(.*?)\s*\]',
    r'colorRegions:\s*\[(.*?)\],',
]

for pattern in patterns:
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f"\nPattern '{pattern}' found match")
        text = match.group(1)
        print(f"Matched text length: {len(text)}")
        print("First 200 chars:", text[:200])
        
        # Try to extract rows
        rows = re.findall(r'\[([^\]]+)\]', text)
        print(f"Found {len(rows)} rows")
        if rows:
            print("First row:", rows[0])
        break
else:
    print("No patterns matched")

# Look for the structure directly
print("\n\nSearching for rows directly...")
rows_match = re.findall(r'\[["A-Z, ]+\]', content)
print(f"Found {len(rows_match)} potential rows")
if rows_match:
    for i, row in enumerate(rows_match[:3]):
        print(f"Row {i}: {row}")
