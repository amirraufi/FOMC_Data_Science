"""
Extract Data from Existing Notebook

This script reads your existing FOMCC (1).ipynb and extracts
the embedded data to CSV files for use with the enhanced framework.

Usage:
    python extract_data_from_notebook.py
"""

import json
import pandas as pd
import re


def extract_dataframes_from_notebook(notebook_path='FOMCC (1).ipynb'):
    """
    Extract dataframes from notebook by executing the cells that create them

    This is a simple extractor - you may need to run your original notebook
    and export manually if this doesn't work.
    """
    print("="*70)
    print("EXTRACTING DATA FROM EXISTING NOTEBOOK")
    print("="*70)

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        print(f"✓ Loaded notebook: {notebook_path}")
        print(f"  Cells: {len(notebook['cells'])}")

        # Look for cells that load or create dataframes
        data_loading_cells = []

        for idx, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])

                # Look for data loading patterns
                if any(pattern in source for pattern in [
                    'read_csv', 'pd.read_csv', 'communications.csv',
                    'gpt_hawk_scores', 'data_with_gpt'
                ]):
                    data_loading_cells.append((idx, source))

        print(f"\n📊 Found {len(data_loading_cells)} cells with data loading code")

        if data_loading_cells:
            print("\nData loading cells found:")
            for idx, source in data_loading_cells[:5]:  # Show first 5
                print(f"\nCell {idx}:")
                print(source[:200] + "..." if len(source) > 200 else source)

        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print("""
Since your data is in the notebook, here's the easiest way to extract it:

1. Open your notebook:
   jupyter notebook "FOMCC (1).ipynb"

2. Run all cells to load your data

3. At the end, add a new cell with:

   # Export data to CSV
   df.to_csv('communications.csv', index=False)
   print(f"✓ Saved {len(df)} documents to communications.csv")

   # If you have the full feature matrix:
   if 'data_with_gpt_bart_finbert' in dir():
       data_with_gpt_bart_finbert.to_csv('data_with_gpt_bart_finbert.csv', index=False)
       print("✓ Saved features to data_with_gpt_bart_finbert.csv")

   # Or whatever your dataframe variables are named:
   # Check with: print([v for v in dir() if 'df' in v.lower()])

4. Run that cell to export your data

5. Then run: python train_models.py

Alternative: If the data files already exist elsewhere on your system,
just copy them to this directory:
   cp /path/to/communications.csv .
   cp /path/to/data_with_gpt_bart_finbert.csv .
        """)

        return True

    except FileNotFoundError:
        print(f"✗ Could not find {notebook_path}")
        print("  Make sure the notebook is in the current directory")
        return False

    except Exception as e:
        print(f"✗ Error reading notebook: {e}")
        return False


def check_existing_csv_files():
    """Check if CSV files already exist"""
    import os

    print("\n" + "="*70)
    print("CHECKING FOR EXISTING DATA FILES")
    print("="*70)

    files_to_check = [
        'communications.csv',
        'data_with_gpt_bart_finbert.csv',
        'gpt_hawk_scores.csv',
        'X_merged.csv',
        'Data_with_gpt_ranks.csv'
    ]

    found_files = []

    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            found_files.append(filename)
            print(f"✓ {filename} - {size:,} bytes")

            # Try to load and show info
            try:
                df = pd.read_csv(filename, nrows=5)
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Preview shape: {df.shape}")
            except Exception as e:
                print(f"  (Could not preview: {e})")
        else:
            print(f"✗ {filename} - not found")

    if found_files:
        print(f"\n✓ Found {len(found_files)} existing data file(s)")
        print("  You may be able to use these directly!")
    else:
        print("\n✗ No existing CSV files found")
        print("  You'll need to export from your notebook")

    return found_files


def create_demo_data():
    """Create demo data for testing"""
    print("\n" + "="*70)
    print("CREATING DEMO DATA (for testing)")
    print("="*70)

    import numpy as np
    from datetime import datetime, timedelta

    # Create demo FOMC data
    print("\nCreating demo communications.csv...")

    dates = pd.date_range('2015-01-01', periods=40, freq='45D')

    texts = []
    for i in range(40):
        inflation = np.random.choice(['elevated', 'moderate', 'easing'])
        labor = np.random.choice(['tight', 'stable', 'softening'])
        growth = np.random.choice(['solid', 'moderate', 'slowing'])

        text = f"""The Federal Reserve has decided to {"maintain" if i % 3 != 0 else "adjust"} the target range for the federal funds rate.
Recent indicators suggest that economic activity has continued at a {growth} pace.
Job gains have been {labor} and the labor market remains strong.
Inflation remains {inflation} and continues to be a concern.
The Committee will continue to monitor the incoming data and adjust policy as appropriate."""

        texts.append(text)

    demo_data = pd.DataFrame({
        'Date': dates,
        'Release Date': dates,
        'Type': ['Statement'] * 40,
        'Text': texts
    })

    demo_data.to_csv('communications_demo.csv', index=False)
    print(f"✓ Created communications_demo.csv with {len(demo_data)} demo statements")
    print("  Use this to test the framework!")

    # Create demo features
    print("\nCreating demo features...")
    demo_features = pd.DataFrame({
        'Date': dates,
        'gpt_score': np.random.randn(40) * 0.5,  # Hawkishness scores
        'bart_score': np.random.randn(40) * 0.5,
        'finbert_pos': np.random.rand(40) * 0.3,
        'finbert_neg': np.random.rand(40) * 0.3,
        'finbert_neutral': np.random.rand(40) * 0.4 + 0.3,
        'hawk_minus_dove': np.random.randn(40) * 2,
        'delta_semantic': np.random.rand(40) * 0.3,
    })

    demo_features.to_csv('demo_features.csv', index=False)
    print(f"✓ Created demo_features.csv with sample NLP features")

    print("\n" + "="*70)
    print("DEMO DATA READY!")
    print("="*70)
    print("""
To test with demo data:

1. Modify train_models.py to use demo files:
   - Change 'communications.csv' to 'communications_demo.csv'
   - Change 'data_with_gpt_bart_finbert.csv' to 'demo_features.csv'

2. Run: python train_models.py

3. This will show you how the framework works

4. Then replace with your real data when ready!

Or just run quick_start_example.py with communications_demo.csv
    """)


def main():
    """Main extraction workflow"""

    # Check for existing files first
    existing_files = check_existing_csv_files()

    # Try to extract from notebook
    extract_dataframes_from_notebook()

    # Offer to create demo data
    print("\n" + "="*70)
    print("OPTIONS")
    print("="*70)
    print("""
What would you like to do?

A) Create demo data to test the framework
B) Extract manually from your notebook (recommended)
C) Skip (you already have the data files)

For option B (recommended):
1. Run: jupyter notebook "FOMCC (1).ipynb"
2. Execute all cells
3. Add export cell (see instructions above)
4. Run: python train_models.py

For option A (quick test):
    """)

    try:
        choice = input("Enter your choice (A/B/C): ").strip().upper()

        if choice == 'A':
            create_demo_data()
        elif choice == 'B':
            print("\n✓ Opening instructions...")
            print("\nIn your notebook, add this cell:")
            print("-" * 70)
            print("""
# Export data for enhanced analysis
import pandas as pd

# Export main communications data
if 'df' in dir():
    df.to_csv('communications.csv', index=False)
    print(f"✓ Saved {len(df)} documents to communications.csv")

# Export feature matrix if it exists
for var_name in dir():
    if 'feature' in var_name.lower() or 'gpt' in var_name.lower():
        var = eval(var_name)
        if isinstance(var, pd.DataFrame) and len(var) > 0:
            filename = f"{var_name}.csv"
            var.to_csv(filename, index=False)
            print(f"✓ Saved {var_name} to {filename}")
            """)
            print("-" * 70)
        elif choice == 'C':
            print("\n✓ Skipping data creation")
            if existing_files:
                print("  You can use the existing files!")
        else:
            print("\n⚠ Invalid choice, skipping...")

    except EOFError:
        print("\n\n(Running in non-interactive mode)")
        print("To create demo data, run:")
        print("  python -c 'from extract_data_from_notebook import create_demo_data; create_demo_data()'")


if __name__ == "__main__":
    main()
