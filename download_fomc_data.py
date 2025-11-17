"""
Download FOMC statements using FedTools
Creates communications.csv with Date and Text columns
"""

import pandas as pd
from fedtools import FOMCStatement

print("=" * 70)
print("DOWNLOADING FOMC STATEMENTS")
print("=" * 70)

print("\nFetching FOMC statements from Federal Reserve...")
try:
    statements = FOMCStatement()
    df = statements.download()

    print(f"✓ Downloaded {len(df)} FOMC statements")

    # Standardize column names
    # FedTools returns columns like 'date', 'statement' or similar
    print(f"\nColumns from FedTools: {list(df.columns)}")

    # Rename columns to match expected format
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    if 'statement' in df.columns:
        df = df.rename(columns={'statement': 'Text'})
    elif 'text' in df.columns:
        df = df.rename(columns={'text': 'Text'})
    elif 'content' in df.columns:
        df = df.rename(columns={'content': 'Text'})

    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Add Type column
    df['Type'] = 'Statement'

    # Keep only necessary columns
    columns_to_keep = ['Date', 'Type', 'Text']
    if 'Release Date' in df.columns:
        columns_to_keep.append('Release Date')

    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Save to CSV
    df.to_csv('communications.csv', index=False)

    print(f"\n✓ Saved to communications.csv")
    print(f"  Date range: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Show sample
    print("\nSample (first statement):")
    print(f"  Date: {df.iloc[0]['Date']}")
    print(f"  Text preview: {df.iloc[0]['Text'][:200]}...")

except Exception as e:
    print(f"\n✗ Error downloading FOMC statements: {e}")
    print("\nTrying alternative approach...")

    # Alternative: Try individual components
    try:
        from fedtools import FOMCMinutes, FOMCPresConference

        print("\nDownloading FOMC Minutes...")
        minutes = FOMCMinutes()
        df_minutes = minutes.download()
        df_minutes['Type'] = 'Minutes'

        print("\nDownloading Press Conferences...")
        press = FOMCPresConference()
        df_press = press.download()
        df_press['Type'] = 'Press Conference'

        # Combine
        df = pd.concat([df_minutes, df_press], ignore_index=True)

        # Standardize columns
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'Text'})

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df.to_csv('communications.csv', index=False)
        print(f"✓ Saved {len(df)} documents to communications.csv")

    except Exception as e2:
        print(f"✗ Alternative approach also failed: {e2}")
        print("\nManual data download required. See DATA_SOURCES.md for alternatives.")

print("\n" + "=" * 70)
