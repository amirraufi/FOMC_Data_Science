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
.gitignore                   # Git ignore rules
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

## Novel Contribution: Change Detection

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

**Why this matters:** Markets react to *surprises*. Changes in Fed language signal policy shifts better than absolute levels.

## Research Roadmap

See `RESEARCH_ROADMAP.md` for detailed publication timeline:
- **Phase 1**: Data & feature engineering ✅
- **Phase 2**: Modeling & validation (in progress)
- **Phase 3**: Robustness checks
- **Phase 4**: Paper writing
- **Phase 5**: Conference & journal submission

Target: 60%+ directional accuracy predicting yield changes

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

### Market Data (Automatically Fetched)
- **FRED** (Federal Reserve Economic Data): Treasury yields, Fed Funds rate
  - DGS2 (2-Year Treasury)
  - DGS5 (5-Year Treasury)
  - DGS10 (10-Year Treasury)
  - DFF (Effective Fed Funds Rate)
- **No API key required** - free via `pandas_datareader`

## Example Usage

```python
from fomc_analysis_utils import FOMCDataLoader, ChangeDetector, MarketReactionCalculator
import pandas as pd

# 1. Load communications
loader = FOMCDataLoader('communications.csv')
statements = loader.load_communications()

# 2. Add change detection features
statements_with_changes = ChangeDetector.add_change_features(statements)

# 3. Fetch market data
market_data = loader.fetch_market_data()

# 4. Calculate market reactions
final_data = MarketReactionCalculator.compute_reactions(
    statements_with_changes,
    market_data,
    horizons=[1, 2]
)

# 5. Train models (see run_analysis.py for complete pipeline)
```

## Expected Results

**Minimum Viable Paper** (Good):
- 55-60% directional accuracy
- Significant improvement over baseline
- Clear SHAP interpretability

**Strong Paper** (Very Good):
- 60-65% directional accuracy
- Economic significance (profitable trading strategy)
- 2-3 compelling case studies

**Exceptional Paper** (Excellent):
- 65%+ directional accuracy
- Additional intraday analysis
- Policy impact potential

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- pandas_datareader (for FRED data)
- transformers (for FinBERT/BART)
- shap (for interpretability)
- See `requirements_enhanced.txt` for full list

## Contributing

This is an academic research project. For questions or collaboration:
1. Check existing documentation in this README
2. Review `RESEARCH_ROADMAP.md` for publication timeline
3. Examine code in `fomc_analysis_utils.py` for implementation details

## License

Academic research project. Please cite if you use this methodology.

## Citation

```bibtex
@article{fomc_nlp_2025,
  title={Predicting Market Reactions to FOMC Communications: A Change Detection Approach},
  author={[Your Names]},
  year={2025},
  note={Working paper}
}
```

## Next Steps

1. **Find data**: Search GitHub/Kaggle for FOMC statements dataset
2. **Run analysis**: `python run_analysis.py`
3. **Generate results**: Review output files and SHAP plots
4. **Start writing**: Use results to draft academic paper
5. **Submit**: Target conferences first (AFA, WFA, NBER), then journals

**Timeline to publication**: 2-3 years (normal for top finance journals)

---

**Questions?**
- Technical implementation → See `fomc_analysis_utils.py`
- Research strategy → See `RESEARCH_ROADMAP.md`
- Just run it → `python run_analysis.py`
