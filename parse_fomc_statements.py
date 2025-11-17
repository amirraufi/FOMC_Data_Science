"""
Parse FOMC statements from the cloned GitHub repository
Creates communications.csv with Date, Type, and Text columns
"""

import pandas as pd
import os
from datetime import datetime

print("=" * 70)
print("PARSING FOMC STATEMENTS FROM REPOSITORY")
print("=" * 70)

statements_dir = 'fomc_statements_repo/statements'

if not os.path.exists(statements_dir):
    print(f"✗ Error: {statements_dir} not found")
    print("  Please run: git clone https://github.com/fomc/statements.git fomc_statements_repo")
    exit(1)

# Get all statement files
statement_files = sorted([f for f in os.listdir(statements_dir) if f.endswith('.txt')])

print(f"\nFound {len(statement_files)} statement files")

# Parse each file
data = []

for filename in statement_files:
    try:
        # Extract date from filename (format: YYYYMMDD.txt)
        date_str = filename.replace('.txt', '')
        date = datetime.strptime(date_str, '%Y%m%d')

        # Read file content
        file_path = os.path.join(statements_dir, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()

        # Skip empty files
        if not text or len(text) < 50:
            print(f"  Skipping {filename} - too short or empty")
            continue

        data.append({
            'Date': date,
            'Release Date': date,  # Same as date for statements
            'Type': 'Statement',
            'Text': text
        })

    except Exception as e:
        print(f"  ✗ Error parsing {filename}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✓ Parsed {len(df)} statements successfully")
print(f"  Date range: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

# Save to CSV
df.to_csv('communications.csv', index=False)
print(f"\n✓ Saved to communications.csv")

# Show statistics
print("\nDataset statistics:")
print(f"  Total statements: {len(df)}")
print(f"  Average text length: {df['Text'].str.len().mean():.0f} characters")
print(f"  Min text length: {df['Text'].str.len().min():.0f} characters")
print(f"  Max text length: {df['Text'].str.len().max():.0f} characters")

# Show sample
print("\nSample (first statement):")
print(f"  Date: {df.iloc[0]['Date']:%Y-%m-%d}")
print(f"  Text preview: {df.iloc[0]['Text'][:300]}...")

print("\nSample (most recent statement):")
print(f"  Date: {df.iloc[-1]['Date']:%Y-%m-%d}")
print(f"  Text preview: {df.iloc[-1]['Text'][:300]}...")

print("\n" + "=" * 70)
print("✓ communications.csv ready for analysis!")
print("=" * 70)
