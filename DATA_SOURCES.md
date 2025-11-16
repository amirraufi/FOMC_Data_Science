# FOMC Data Sources - Actual Datasets

This document lists actual, available datasets for FOMC communications and market data.

## FOMC Text Data (Statements, Minutes, Transcripts)

### Top Recommended Sources:

#### 1. **GitHub: fomc/statements** ⭐ BEST FOR STATEMENTS
- **URL**: https://github.com/fomc/statements
- **Coverage**: 1994-present
- **Content**: All FOMC Policy Statements and Implementation Notes as text files
- **Format**: Plain text files, organized by year
- **Quality**: Official, complete, easy to parse
- **Use case**: Perfect for your 2000-2025 analysis

#### 2. **Hugging Face: gtfintechlab/fomc_communication** ⭐ BEST COMPREHENSIVE
- **URL**: https://huggingface.co/datasets/gtfintechlab/fomc_communication
- **Coverage**: 214 meeting minutes, 1,026 speeches, 63 press conference transcripts
- **Content**: Tokenized and annotated with hawkish/dovish labels
- **Format**: Pre-processed, ready for NLP
- **Quality**: Academic quality (Georgia Tech researchers)
- **License**: CC BY-NC 4.0
- **Use case**: Best for comprehensive analysis with pre-labeled sentiment

#### 3. **Kaggle: Federal Reserve FOMC Minutes & Statements**
- **URL**: https://www.kaggle.com/datasets/drlexus/fed-statements-and-minutes
- **Coverage**: Through April 2023
- **Content**: FOMC statements and minutes
- **Format**: CSV/structured
- **Quality**: Good, community-validated
- **Use case**: Quick start, already in tabular format

#### 4. **GitHub: David-Woroniuk/FedTools** ⭐ BEST PYTHON LIBRARY
- **URL**: https://github.com/David-Woroniuk/FedTools
- **Type**: Python library for extracting Fed data
- **Features**:
  - `fedtools.FOMCStatement()` - Download all statements
  - `fedtools.FOMCMinutes()` - Download all minutes
  - `fedtools.BeigeBook()` - Download Beige Books
- **Format**: Returns Pandas DataFrames
- **Use case**: Easiest programmatic access

```python
# Example using FedTools
from fedtools import FOMCStatement

# Download all FOMC statements
statements = FOMCStatement()
df = statements.download()  # Returns DataFrame with dates and text
```

#### 5. **GitHub: vtasca/fed-statement-scraping**
- **URL**: https://github.com/vtasca/fed-statement-scraping
- **Features**: Automatically scrapes FOMC statements and minutes
- **Format**: Automated scraper, keeps data updated
- **Use case**: If you want to maintain fresh data

### Additional Text Sources:

- **jm4474/FOMCTextAnalysis**: https://github.com/jm4474/FOMCTextAnalysis
  - Tools for scraping and extracting FOMC documents

- **yukit-k/centralbank_analysis**: https://github.com/yukit-k/centralbank_analysis
  - Multi-central bank analysis (FOMC, ECB, BOE, BOJ)
  - Includes market data integration from Quandl

- **Federal Reserve Governors Speeches (Kaggle)**:
  - https://www.kaggle.com/datasets/natanm/federal-reserve-governors-speeches-1996-2020
  - Coverage: 1996-2020
  - Useful for studying individual Fed member communications

---

## Market Data (Treasury Yields, Fed Funds Futures)

### Primary Source: FRED (Federal Reserve Economic Data) ⭐ OFFICIAL

#### Direct Access (Best):
- **URL**: https://fred.stlouisfed.org/
- **API**: Free, no key required with `pandas_datareader`
- **Coverage**: 1960s-present (depends on series)
- **Quality**: Official government data

**Key Series for FOMC Analysis:**
```python
import pandas_datareader.data as web
from datetime import datetime

# Treasury yields
DGS2  = '2-Year Treasury Constant Maturity Rate'
DGS5  = '5-Year Treasury Constant Maturity Rate'
DGS10 = '10-Year Treasury Constant Maturity Rate'

# Fed Funds
DFF   = 'Effective Federal Funds Rate'

# Fetch example
data = web.DataReader(['DGS2', 'DGS5', 'DGS10', 'DFF'],
                      'fred',
                      start='2000-01-01',
                      end='2025-12-31')
```

#### Python Library Access:
- **GitHub: mortada/fredapi**: https://github.com/mortada/fredapi
  - Full-featured FRED API wrapper
  - More control than pandas_datareader

### Fed Funds Futures:

#### CME FedWatch Tool
- **URL**: https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html
- **Content**: Probabilities of Fed rate changes implied by Fed Funds futures
- **Format**: Web-based tool (requires scraping or manual download)
- **Use case**: Market expectations before FOMC meetings

#### GitHub: fomc-future-predictor
- **URL**: https://github.com/stock-market-predictor/fomc-future-predictor
- **Content**: Analysis of 30-day Fed Funds futures vs FOMC decisions
- **Use case**: Example implementation of futures-based analysis

---

## Intraday Market Data (High-Frequency)

### Challenge:
True intraday data (minute-by-minute or tick-by-tick) for Treasury markets is **not freely available** for historical periods (2000-2020). This is proprietary data sold by:
- Bloomberg Terminal ($$$)
- Refinitiv/Reuters ($$$)
- QuantConnect (limited free tier)
- Interactive Brokers (recent data only)

### Workarounds:

#### 1. **Yahoo Finance (yfinance)** - Limited Free Intraday
- **Coverage**: Last 60 days only for 5-minute data
- **Tickers**: TLT (20Y Treasury ETF), IEF (7-10Y), SHY (1-3Y)
- **Limitation**: Cannot get historical intraday back to 2000-2020

```python
import yfinance as yf

# Recent event only (last 60 days)
tlt = yf.Ticker("TLT")
intraday = tlt.history(period="5d", interval="5m")
```

#### 2. **Academic Data Sources**
- Check if your institution has access to:
  - WRDS (Wharton Research Data Services) - Has TAQ (Trade and Quote) data
  - Bloomberg Terminal access
  - Refinitiv/Reuters access

#### 3. **Alternative Approach** (Recommended for Your Paper)
Instead of true intraday data, use:
- **Daily data for 2000-2020** (complete coverage via yfinance/FRED)
- **Case studies with intraday** for 2-3 recent events
- Focus on change detection contribution rather than high-frequency analysis

This is academically acceptable - many published papers use daily data.

---

## Recommended Data Setup for Your Project

### Minimal Setup (Fastest):
```bash
# 1. Install FedTools
pip install fedtools

# 2. Download statements
from fedtools import FOMCStatement
statements = FOMCStatement().download()
statements.to_csv('communications.csv', index=False)

# 3. Fetch market data (your existing code already does this)
# Uses pandas_datareader to get FRED data
```

### Comprehensive Setup (Best for Publication):
1. **Text Data**: Use Hugging Face `gtfintechlab/fomc_communication`
   - Most comprehensive
   - Pre-annotated sentiment
   - Academic quality

2. **Market Data**: Use FRED via `pandas_datareader`
   - Daily Treasury yields (DGS2, DGS5, DGS10)
   - Fed Funds rate (DFF)
   - Completely free, official data

3. **Your Novel Contribution**: Change detection features
   - Your existing `ChangeDetector` class
   - This is what makes your paper unique!

4. **Existing NLP Features**: Your current work
   - GPT-4, FinBERT, BART scores
   - Keep `data_with_gpt_bart_finbert.csv`

### File Structure:
```
communications.csv               # From FedTools or Hugging Face
data_with_gpt_bart_finbert.csv  # Your existing NLP features
market_data/                     # Auto-fetched from FRED
  ├── treasury_yields.csv
  └── fed_funds.csv
```

---

## Data Quality Comparison

| Source | Coverage | Quality | Format | Cost | Best For |
|--------|----------|---------|--------|------|----------|
| **fomc/statements** (GitHub) | 1994-present | ⭐⭐⭐⭐⭐ | Text files | Free | Statements only |
| **gtfintechlab/fomc_communication** (HF) | 214 min + 1,026 speeches | ⭐⭐⭐⭐⭐ | Annotated | Free | Comprehensive |
| **FedTools** (Python) | 1994-present | ⭐⭐⭐⭐⭐ | DataFrame | Free | Easiest access |
| **Kaggle FOMC** | Through 2023 | ⭐⭐⭐⭐ | CSV | Free | Quick start |
| **FRED** (Market data) | 1960s-present | ⭐⭐⭐⭐⭐ | CSV/API | Free | Official govt data |
| **Yahoo Finance** | Daily: 2000+, Intraday: 60d | ⭐⭐⭐ | API | Free | Recent data |
| **Bloomberg/Reuters** | Full intraday history | ⭐⭐⭐⭐⭐ | Proprietary | $$$ | Not feasible |

---

## What I Recommend You Use

Based on your academic publication goals:

### Phase 1: Get the data (This week)
1. **Install FedTools**: `pip install fedtools`
2. **Download statements**:
   ```python
   from fedtools import FOMCStatement
   df = FOMCStatement().download()
   ```
3. **Already have market data**: Your code fetches from FRED ✓

### Phase 2: Enrich (Optional)
- Download Hugging Face dataset for pre-labeled sentiment scores
- Compare with your GPT-4 scores (great robustness check!)

### Phase 3: Focus on your contribution
- Change detection is your novelty
- Daily market reactions are fine (most papers use daily)
- Don't waste time chasing intraday data you can't get

---

## Papers Using These Data Sources

Reference these in your literature review:

1. **Hansen, McMahon & Prat (2018)** - "Transparency and Deliberation Within the FOMC"
   - Used FOMC transcripts and statements
   - Published in Journal of Finance

2. **Lucca & Trebbi (2009)** - "Measuring Central Bank Communication"
   - Used FOMC statements
   - Published in Journal of Monetary Economics

3. **Bybee et al. (2021)** - "Business News and Business Cycles"
   - Used structured text analysis
   - Published in Journal of Finance

Your contribution: **First to use comprehensive change detection across full FOMC statement history with multi-modal NLP**

---

## Quick Links Reference

**FOMC Text:**
- https://github.com/fomc/statements
- https://huggingface.co/datasets/gtfintechlab/fomc_communication
- https://github.com/David-Woroniuk/FedTools

**Market Data:**
- https://fred.stlouisfed.org/
- https://github.com/mortada/fredapi

**Kaggle:**
- https://www.kaggle.com/datasets/drlexus/fed-statements-and-minutes

---

## Last Updated
2025-01-16 (based on web search results)

Use these sources instead of trying to build custom scrapers or chase unavailable intraday data.
