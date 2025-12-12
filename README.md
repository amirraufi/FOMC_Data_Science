# FOMC Market Reaction Analysis

Predicting market reactions to Federal Reserve (FOMC) communications using multi-modal NLP and novel change detection features.

## Overview

This project analyzes FOMC statements (2000-2025) to predict Treasury market reactions using:
- **Multi-modal NLP**: GPT-4, FinBERT, BART sentiment analysis
- **Novel Change Detection**: 30+ features tracking linguistic shifts between consecutive statements
- **Market Data**: Treasury yields (FRED), Fed Funds futures
- **Interpretability**: SHAP analysis for feature importance

**Target**: Publication in top finance journals (Journal of Finance, JFE, RFS)

## Core Files

```
fomc_analysis_utils.py       # Main utility library (ChangeDetector, data loaders, market calculators)
run_analysis.py              # Complete training pipeline with time-series CV and SHAP analysis
FOMC_Enhanced_Research.ipynb # Research notebook
requirements_enhanced.txt    # Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 2. Prepare Data Files

You need to provide:
- `communications.csv` - FOMC statements with columns: Date, Release Date, Type, Text
- `data_with_gpt_bart_finbert.csv` - (Optional) Your existing NLP features

**Finding Data Sources:**
- Search GitHub for "FOMC statements dataset" or "Federal Reserve communications"
- Check Kaggle for "FOMC" or "Fed minutes" datasets
- FRED (Federal Reserve Economic Data) for market data - fetched automatically

### 3. Run Complete Analysis
```bash
python run_analysis.py
```

This will:
- Load FOMC communications
- Generate change detection features (novel contribution!)
- Fetch market data from FRED
- Calculate market reactions (1-day, 2-day changes)
- Train models with time-series cross-validation
- Run SHAP analysis for interpretability
- Generate results files

### 4. Output Files
```
model_results.csv           # Model performance metrics
feature_importance.csv      # SHAP-based feature rankings
shap_summary_plot.png       # Feature importance visualization
predictions_validation.csv  # Predictions on validation set
predictions_holdout.csv     # Predictions on 2024-2025 holdout
```

##  Change Detection

The key innovation is **change detection** - comparing consecutive FOMC statements to capture linguistic shifts:

```python
from fomc_analysis_utils import ChangeDetector

# Detect changes between consecutive statements
changes = ChangeDetector.detect_changes(current_text, previous_text)

# Returns 30+ features:
# - Sentences added/removed/modified
# - Key phrase changes (inflation, rates, labor, growth)
# - Semantic similarity scores
# - Language intensity shifts
```


## Key Features

### Change Detection (30+ features)
- Sentence-level diffs (added/removed/unchanged)
- Key phrase tracking across statements
- Semantic similarity measures
- Language intensity changes

### Market Reactions
- 1-day, 2-day Treasury yield changes (basis points)
- Fed Funds futures reactions
- Multiple maturities (2Y, 5Y, 10Y)

### Modeling Approach
- Time-series cross-validation (no look-ahead bias)
- Multiple model families (Linear, RF, GBM, Neural Nets)
- SHAP interpretability analysis
- 2024-2025 holdout test set

## Data Sources

### FOMC Communications
You must provide this file:
- **File**: `communications.csv`
- **Columns**: Date, Release Date, Type, Text
- **Where to find**:
  - Search GitHub: "FOMC statements" OR "Federal Reserve communications dataset"
  - Check Kaggle: "FOMC" OR "Fed minutes"
  - Manual collection: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm



