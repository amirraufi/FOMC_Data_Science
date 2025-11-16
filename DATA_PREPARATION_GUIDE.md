# Data Preparation Guide

## Required Data Files

To run the enhanced analysis, you need these data files:

### 1. `communications.csv` (REQUIRED)
Your FOMC communications data with these columns:
- `Date`: Date of FOMC meeting (datetime)
- `Release Date`: When statement was released (datetime)
- `Type`: "Statement" or "Minute"
- `Text`: Full text of the communication

**Where to get it:**
- You already have this data (it's referenced in your original notebook)
- Should be in your working directory or data folder
- If you need to download again: [Federal Reserve website](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)

**Format check:**
```python
import pandas as pd
df = pd.read_csv('communications.csv')
print(df.head())
print(df.columns)
# Should show: ['Date', 'Release Date', 'Type', 'Text']
```

### 2. `data_with_gpt_bart_finbert.csv` (OPTIONAL but recommended)
Your existing NLP features from the original analysis:
- GPT-4 hawkishness scores
- BART classification scores
- FinBERT sentiment scores
- Any other features you computed

**This file is optional** - if you don't have it, the code will use only change detection features.

---

## Quick Data Check

Run this to verify your data is ready:

```python
import pandas as pd
import os

# Check required file
if os.path.exists('communications.csv'):
    df = pd.read_csv('communications.csv')
    print(f"✓ communications.csv found: {len(df)} documents")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
else:
    print("✗ communications.csv NOT FOUND")
    print("  Please add this file to run the analysis")

# Check optional file
if os.path.exists('data_with_gpt_bart_finbert.csv'):
    features = pd.read_csv('data_with_gpt_bart_finbert.csv')
    print(f"\n✓ data_with_gpt_bart_finbert.csv found: {features.shape}")
    print(f"  Columns: {features.columns.tolist()[:10]}...")
else:
    print("\n⚠ data_with_gpt_bart_finbert.csv NOT FOUND")
    print("  Will use only change detection features")
```

---

## Extracting Data from Your Existing Notebook

If your data is embedded in `FOMCC (1).ipynb`, here's how to extract it:

### Option 1: Run the notebook and export
```python
# In your existing notebook, after loading data:
df.to_csv('communications.csv', index=False)
print(f"✓ Saved {len(df)} documents to communications.csv")

# If you have the full feature matrix:
full_features.to_csv('data_with_gpt_bart_finbert.csv', index=False)
print(f"✓ Saved features to data_with_gpt_bart_finbert.csv")
```

### Option 2: Convert notebook to script
```bash
# Install nbconvert if needed
pip install nbconvert

# Convert notebook to Python script
jupyter nbconvert --to script "FOMCC (1).ipynb"

# This creates "FOMCC (1).py" which you can run or extract data from
```

---

## Alternative: Use Demo Data (for testing)

If you want to test the framework before getting your full data, create a small demo dataset:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create demo FOMC data
dates = pd.date_range('2020-01-01', periods=20, freq='3M')
demo_data = {
    'Date': dates,
    'Release Date': dates,
    'Type': ['Statement'] * 20,
    'Text': [
        f"The Federal Reserve has decided to maintain the target range. "
        f"Inflation remains {'elevated' if i % 2 == 0 else 'moderate'}. "
        f"The labor market is {'tight' if i % 3 == 0 else 'stable'}. "
        f"Economic growth is {'solid' if i % 2 == 0 else 'slowing'}."
        for i in range(20)
    ]
}

df = pd.DataFrame(demo_data)
df.to_csv('communications_demo.csv', index=False)
print("✓ Created demo data: communications_demo.csv")
print("  Use this to test the framework, then replace with real data")
```

---

## Data Format Requirements

### communications.csv

**Required columns:**
- `Date`: YYYY-MM-DD format
- `Release Date`: YYYY-MM-DD format
- `Type`: Must be "Statement" or "Minute"
- `Text`: String (can be long)

**Example:**
```csv
Date,Release Date,Type,Text
2023-03-22,2023-03-22,Statement,"The Federal Reserve has decided to raise the target range for the federal funds rate to 4.75 to 5 percent. Recent indicators suggest that economic activity has continued to expand at a modest pace..."
2023-05-03,2023-05-03,Statement,"The Federal Reserve has decided to raise the target range for the federal funds rate to 5 to 5.25 percent. The U.S. banking system is sound and resilient..."
```

### data_with_gpt_bart_finbert.csv (optional)

**Recommended columns:**
- `Date`: YYYY-MM-DD (to merge with communications.csv)
- `gpt_score`: GPT-4 hawkishness score
- `bart_score`: BART classification score
- `finbert_pos`, `finbert_neg`, `finbert_neutral`: FinBERT sentiments
- `hawk_minus_dove`: Net hawkishness from keyword counts
- `delta_semantic`: Semantic similarity score
- Any other features you computed

**The code will automatically use any columns starting with:**
- `gpt_`
- `bart_`
- `finbert_`
- Plus specific columns like `hawk_minus_dove`, `delta_semantic`

---

## Once You Have the Data

### Step 1: Verify data is in place
```bash
ls -lh *.csv
# Should show:
#   communications.csv
#   data_with_gpt_bart_finbert.csv (optional)
```

### Step 2: Run the quick start
```bash
python quick_start_example.py
```

This will:
- Load your data
- Add change detection features
- Compute market reactions
- Show summary statistics
- Generate visualizations

### Step 3: Train models
```bash
python train_models.py
```

This will:
- Merge all features
- Train multiple models
- Run SHAP analysis
- Generate publication figures
- Save results to CSV

### Step 4: Review results
Check these generated files:
- `model_results.csv`: Model performance comparison
- `feature_importance.csv`: Which features matter most
- `model_comparison.png`: Performance visualization
- `shap_summary_plot.png`: Feature importance plot
- `shap_bar_plot.png`: Feature importance bar chart

---

## Troubleshooting

### "FileNotFoundError: communications.csv"
**Solution**: Make sure the file is in the same directory as your Python scripts
```bash
pwd  # Check current directory
ls communications.csv  # Verify file exists
```

### "KeyError: 'Date'"
**Solution**: Your CSV might have different column names
```python
# Check what columns you actually have:
df = pd.read_csv('communications.csv')
print(df.columns)

# Rename if needed:
df = df.rename(columns={'date': 'Date', 'release_date': 'Release Date'})
df.to_csv('communications.csv', index=False)
```

### "No such file or directory"
**Solution**: Specify full path to data file
```python
# Instead of:
loader = FOMCDataLoader('communications.csv')

# Use:
loader = FOMCDataLoader('/full/path/to/communications.csv')
```

### Market data fetch fails
**Solution**: FRED API might be rate-limited
```python
# Add delay between requests
import time
time.sleep(1)  # Wait 1 second between API calls
```

Or download data manually from:
- https://fred.stlouisfed.org/series/DFF
- https://fred.stlouisfed.org/series/DGS2
- https://fred.stlouisfed.org/series/DGS5
- https://fred.stlouisfed.org/series/DGS10

---

## Data Sources (if you need to rebuild)

### FOMC Statements and Minutes
- **Source**: [Federal Reserve](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- **Alternative**: [FRASER](https://fraser.stlouisfed.org/title/677) (historical archive)

### Market Data
- **Treasury Yields**: [FRED](https://fred.stlouisfed.org/)
  - DGS2, DGS5, DGS10 (daily)
- **Fed Funds Rate**: [FRED DFF](https://fred.stlouisfed.org/series/DFF)
- **Alternative**: [Yahoo Finance](https://finance.yahoo.com/) for treasury ETFs (TLT, IEF, SHY)

### NLP Features (if regenerating)
- **GPT-4**: OpenAI API (requires API key)
- **FinBERT**: HuggingFace `ProsusAI/finbert`
- **BART**: HuggingFace `facebook/bart-large-mnli`
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

---

## Summary Checklist

Before running the analysis:

- [ ] `communications.csv` exists and has required columns
- [ ] Data covers 2000-2025 period (or your chosen range)
- [ ] `data_with_gpt_bart_finbert.csv` exists (optional)
- [ ] Dependencies installed: `pip install -r requirements_enhanced.txt`
- [ ] Can import utilities: `from fomc_analysis_utils import FOMCDataLoader`

Ready to go:
- [ ] Run `python quick_start_example.py` successfully
- [ ] Run `python train_models.py` successfully
- [ ] Results files generated (CSV, PNG)

Then you're ready to:
- [ ] Analyze results
- [ ] Write your paper
- [ ] Submit to journal!

---

**Need help?** Check:
- `ENHANCED_README.md` for overall usage
- `RESEARCH_ROADMAP.md` for research plan
- `IMPLEMENTATION_SUMMARY.md` for what was built

Good luck with your research! 📊🚀
