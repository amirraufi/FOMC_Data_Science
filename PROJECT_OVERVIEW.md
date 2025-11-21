# FOMC Data Science Research Project - Comprehensive Overview

## Executive Summary

This is an **academic research project** predicting Treasury market reactions to Federal Reserve (FOMC) communications using **multi-modal NLP** combined with novel **change detection** features. The project targets top-tier finance journals (Journal of Finance, JFE, RFS) with a goal of 60%+ directional accuracy in predicting yield changes.

---

## 1. MAIN RESEARCH QUESTION & OBJECTIVE

### Primary Question
**Can we predict market reactions to FOMC communications using NLP-extracted linguistic features, particularly changes between consecutive statements?**

### Key Innovation: Change Detection
Rather than analyzing statements in isolation, the project's main innovation is **comparing consecutive FOMC statements** to capture linguistic shifts that signal policy changes. This addresses a fundamental insight: **markets react to surprises, not absolute levels**.

### Specific Objectives
- Develop **30+ change detection features** capturing sentence-level and word-level linguistic shifts
- Integrate multi-modal NLP (GPT-4, FinBERT, BART) with change detection
- Achieve **60%+ directional accuracy** predicting 1-day Treasury yield changes
- Provide **SHAP-based interpretability** to explain which features matter most
- Create a **diagnostic system** that acknowledges uncertainty (percentile scoring, probabilistic forecasts)

### Target Outcomes
- **Minimum viable paper**: 55-60% accuracy, significant improvement over baseline
- **Strong paper**: 60-65% accuracy, economic significance (profitable trading strategy)
- **Exceptional paper**: 65%+ accuracy, full intraday analysis, policy impact

---

## 2. DATA SOURCES & DATASET

### FOMC Communications Data
- **File**: `communications.csv` (12 MB)
- **Records**: ~432 FOMC statements from 2000-2025
- **Content**: 
  - Date, Release Date, Type, Full statement text
  - Types: Statements and Minutes
  - Currently filtered to **Statements only** (excluding Minutes)
- **Coverage**: 25 years of Federal Reserve communications

### Market Data (Treasury Yields)
**Source**: FRED (Federal Reserve Economic Data) - automatically fetched
- **DGS2**: 2-Year Treasury Yield
- **DGS5**: 5-Year Treasury Yield
- **DGS10**: 10-Year Treasury Yield
- **DFF**: Effective Federal Funds Rate
- **Collection**: Automatic via `pandas_datareader` (no API key required)
- **Frequency**: Daily
- **Time Horizons**: 1-day and 2-day changes (basis points)

### Enhanced Dataset
- **File**: `data_with_gpt_bart_finbert.csv` (81 KB)
- **Records**: 432 statements with 24 initial features
- **Features** (initial):
  - `finbert_score`: FinBERT sentiment (-1 to 1)
  - `hawk_cnt`, `dove_cnt`: Keyword counts
  - `gpt_hawk_score`: GPT-4 hawkishness (-2 to 2)
  - `bart_score`: BART probability
  - `cos_prev`: Cosine similarity to previous statement
  - `delta_semantic`: Semantic change from previous
  - Market reactions: `dy2_1d_bp`, `dy5_1d_bp`, `dy10_1d_bp` (yield changes in basis points)

---

## 3. ANALYSES & METHODOLOGIES

### Phase 1: Multi-Modal NLP Feature Extraction (COMPLETED)

#### 1.1 GPT-4 Hawkishness Scoring
- Uses OpenAI GPT-4 API
- Scores statements from -2 (very dovish) to +2 (very hawkish)
- Includes reasoning (available in `gpt_reason` column)
- Captures overall policy stance interpretation

#### 1.2 FinBERT Sentiment Analysis
- Pretrained transformer model from `transformers` library
- Outputs: sentiment score, probability of positive/negative/neutral
- Specifically trained on financial text
- Captures sentiment at different granularities

#### 1.3 BART Zero-Shot Classification
- BART model for topic classification
- Classifies statements as "hawkish" or "dovish"
- Provides confidence scores (`bart_hawk_prob`)

#### 1.4 Semantic Embeddings
- Cosine similarity to previous statement (`cos_prev`)
- Semantic delta from previous statement (`delta_semantic`)
- Captures overall textual continuity

#### 1.5 Keyword Analysis
- Manual hawk/dove keyword extraction
- `hawk_cnt`: Counts of hawkish indicators
- `dove_cnt`: Counts of dovish indicators
- `hawk_minus_dove`: Net hawkish signal

### Phase 2: NOVEL - Word-Level & Sentence-Level Change Detection (COMPLETED)

#### 2.1 Sentence-Level Changes (32 features)
Compares consecutive statements at the sentence level:
- `change_sentences_added`: New sentences in current statement
- `change_sentences_removed`: Removed sentences
- `change_sentences_unchanged`: Unchanged sentences
- `change_overall_similarity`: Text similarity (SequenceMatcher ratio)
- `change_text_length_pct`: Percentage change in statement length
- `change_sentence_count`: Change in number of sentences
- **Key phrase tracking**: Detection of specific policy-related phrases:
  - Inflation language (elevated, moderating, easing)
  - Rate language (increases, cuts, hold)
  - Forward guidance (data-dependent, patient)
  - Labor market (tight, softening)
  - Economic outlook (solid growth, slowing)

#### 2.2 Word-Level Linguistic Features (24 new features)

**SubtleLinguisticAnalyzer** class extracts:

##### Hedge vs. Certainty Language
- HEDGE_WORDS: {may, might, could, possibly, likely, etc.}
- CERTAINTY_WORDS: {will, shall, must, certainly, definitely, etc.}
- Features: current counts, change from previous, ratio

**Why this matters**: "may raise rates" vs "will raise rates" signals different Fed confidence

##### Fed-Specific Word Substitutions
Tracks intensity of key policy words across categories:

1. **Inflation Duration**:
   - transitory (intensity 1) → temporary (2) → persistent (3)
   - Example: "persistent" inflation = more hawkish than "transitory"

2. **Policy Stance**:
   - patient (1) → data-dependent (2) → expeditious (3)

3. **Rate Path Speed**:
   - gradual (1) → measured (2) → rapid (3)

4. **Policy Urgency**:
   - appropriate (1) → warranted (2) → necessary (3)

5. **Inflation Level**:
   - moderating (1) → elevated (2) → high (3) → very high (4)

##### Adjective Intensity
- INTENSIFIERS: {very, highly, extremely, significantly}
- DIMINISHERS: {somewhat, slightly, moderately, relatively}
- Tracks change from previous statement
- "Inflation is elevated" vs "Inflation is VERY elevated"

##### Negation Changes
- Tracks addition/removal of negations (not, no, never, none)
- Meaning reversal is significant in Fed language

##### Verb Tense Changes
- Future tense (will): forward guidance
- Present tense (is): current assessment
- Past tense (was): backward-looking
- Changes in tense distribution signal policy focus shift

### Phase 3: Market Reaction Calculation

#### Targets (Dependent Variables)
For each FOMC release date:
- **1-day change** (release date to next trading day)
- **2-day change** (release date to 2 business days later)

Yields measured in **basis points** (1 bp = 0.01%)

#### Cross-Sectional Analysis
Can analyze reactions across:
- Different Treasury maturities (2Y, 5Y, 10Y)
- Yield curve shape (2s10s spread)
- Fed Funds futures (policy expectations)

### Phase 4: Time-Series Modeling (IN PROGRESS)

#### Train/Validation/Holdout Splits
- **Training**: Pre-2017 (~160 statements)
- **Validation**: 2017-2023 (~140 statements)
- **Holdout**: 2024-2025 (~130 statements) - true out-of-sample test

**Why this split matters**: Avoids look-ahead bias. Holdout tests on truly future data.

#### Model Families Implemented
1. **Linear Models**:
   - OLS Regression
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)

2. **Tree-Based Models**:
   - Random Forest (100 trees)
   - Gradient Boosting Machines

3. **Deep Learning** (mentioned):
   - Neural Networks (MLP)
   - LSTM for sequence analysis (future)

#### Performance Metrics
- **Directional Accuracy**: % of sign predictions correct (primary metric)
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Explained variance

#### Current Best Model
- **Model**: Random Forest
- **Performance**: CV RMSE 7.61 bp, Holdout RMSE 6.68 bp
- **Features**: 99 numerical features (NLP + changes)

### Phase 5: Interpretability - SHAP Analysis

#### SHAP (SHapley Additive exPlanations)
Uses game theory to decompose model predictions:
- Each feature gets an "importance" score
- Shows which features matter most for predictions
- Enables human interpretation of black-box models

#### Outputs Generated
- **Summary plots**: Top 20 features by importance
- **Force plots**: Individual prediction explanations
- **Dependence plots**: Feature value → prediction relationship
- **Feature rankings CSV**: All features ranked

#### Key Finding
- **Top word-level feature**: `subtle_certainty_word_count_current` (rank #8)
- Shows word-level features are competitive with sentence-level features

### Phase 6: Diagnostic System (NEW - STREAMLIT APP)

Rather than point predictions, provides:

1. **Percentile Scoring**: How hawkish relative to all history (0-100%)
2. **Change Highlighting**: What linguistic shifts were detected
3. **Nearest Neighbors**: Similar historical episodes + their outcomes
4. **Probabilistic Forecast**: Conditional distribution (10th/50th/90th percentiles)

---

## 4. KEY FINDINGS & OUTPUTS

### Current State: In Progress

**Completed:**
- ✅ Data loading and integration
- ✅ All NLP features extracted
- ✅ Change detection (sentence + word level)
- ✅ Market reaction calculations
- ✅ Model training (Random Forest, Ridge, Lasso, GBM)
- ✅ SHAP interpretability analysis
- ✅ Diagnostic system implementation
- ✅ Streamlit app for interactive analysis

**In Progress:**
- Model evaluation on holdout 2024-2025 data
- Paper writing

**Planned:**
- Robustness checks (alternative specifications, different horizons)
- Conference submissions (AFA, WFA, NBER)
- Journal submissions (Journal of Finance, JFE, RFS)

### Performance Benchmarks

| Metric | Target | Status |
|--------|--------|--------|
| Directional Accuracy (DA) | 60%+ | Testing |
| Baseline DA | ~53% (always correct) | Documented |
| CV RMSE (Random Forest) | <8 bp | **7.61 bp** ✓ |
| Holdout RMSE | <8 bp | **6.68 bp** ✓ |
| Top Features Identified | Yes | **SHAP analysis done** ✓ |

### Key Finding: Change Detection Matters
- Word-level feature `subtle_certainty_word_count_current` ranks #8 in importance
- Sentence-level changes consistently outperform statement-level features
- **Markets react to changes in Fed language, not absolute levels**

---

## 5. NOTEBOOKS, SCRIPTS & DOCUMENTATION

### Main Files (4,300+ lines of code)

#### Core Research
- **`FOMC_Enhanced_Research.ipynb`** (931 lines)
  - Main research notebook
  - Exploratory data analysis
  - Feature engineering pipeline
  - Model training & evaluation
  - SHAP analysis

#### Utilities Library
- **`fomc_analysis_utils.py`** (958 lines)
  - `FOMCDataLoader`: Data loading & market data fetching
  - `MarketReactionCalculator`: Reaction computation
  - `ChangeDetector`: Sentence-level change detection
  - `SubtleLinguisticAnalyzer`: Word-level feature extraction
  - `DiagnosticAnalyzer`: Percentile scoring & change highlighting
  - `ProbabilisticPredictor`: Conditional distributions
  - `TimeSeriesSplitter`: Proper train/val/holdout splits
  - `ModelEvaluator`: Performance evaluation

#### Analysis Pipeline
- **`run_analysis.py`** (557 lines) - MAIN PIPELINE
  - Step 1: Load existing NLP features
  - Step 2: Add sentence & word-level change detection
  - Step 3: Fetch fresh FRED data
  - Step 4: Train models with time-series CV
  - Step 5: Run SHAP analysis
  - Output files: `data_enhanced_with_changes.csv`, model results, feature importance

- **`run_shap_analysis.py`** (131 lines)
  - Dedicated SHAP analysis script
  - Feature importance rankings

#### Visualization & Output
- **`create_plots.py`** (493 lines)
  - Publication-quality plots
  - EDA visualizations
  - Model comparison plots
  - Feature importance visualizations
  - Case study plots
  - Output folder: `plots/`

#### Interactive Web App
- **`app_streamlit_diagnostic.py`** (625 lines)
  - Streamlit web interface
  - Diagnostic analysis tool
  - Historical data explorer
  - Model performance dashboard
  - Runs with: `streamlit run app_streamlit_diagnostic.py`

#### Testing
- **`test_diagnostic_system.py`** (213 lines)
  - Tests diagnostic analysis components
  - Percentile scoring validation
  - Change highlighting tests
  - Nearest neighbor retrieval tests
  - Probabilistic prediction tests

- **`test_different_horizons.py`** (278 lines)
  - Tests alternative prediction horizons
  - Multi-target evaluation

- **`test_subtle_features.py`** (140 lines)
  - Tests word-level linguistic features
  - Feature extraction validation

### Documentation

#### Main README
- **`README.md`** (6.8 KB)
  - Project overview
  - Quick start guide
  - Data preparation instructions
  - Expected results benchmarks
  - Citation format

#### Research Roadmap
- **`RESEARCH_ROADMAP.md`** (15 KB)
  - Comprehensive publication timeline
  - 6 phases from data to publication
  - Weekly checklists
  - Paper structure & section guidance
  - Conference & journal submission strategy
  - Risk mitigation strategies
  - Success metrics for different paper quality levels

#### System Flowchart
- **`DIAGNOSTIC_FLOWCHART.md`** (11.7 KB)
  - 7 Mermaid diagrams showing system architecture
  - Feature pipeline detail
  - Diagnostic analysis workflow
  - Streamlit app flow
  - Data flow architecture
  - Before/after comparison (point vs diagnostic)
  - Word-level feature extraction detail

#### Streamlit Documentation
- **`STREAMLIT_README.md`** (6.5 KB)
  - Interactive app guide
  - Feature explanations
  - Usage instructions
  - Customization options
  - Deployment instructions
  - Troubleshooting

---

## 6. PROJECT STRUCTURE & FILE ORGANIZATION

```
/home/user/FOMC_Data_Science/
├── Data Files
│   ├── communications.csv (12 MB)          [432 FOMC statements with text]
│   ├── data_with_gpt_bart_finbert.csv (81 KB)  [NLP features extracted]
│   └── data_enhanced_with_changes.csv      [Generated by run_analysis.py - all features]
│
├── Main Analysis Pipeline
│   ├── run_analysis.py (557 lines)         [MAIN SCRIPT - orchestrates entire pipeline]
│   ├── fomc_analysis_utils.py (958 lines)  [Utility classes]
│   ├── FOMC_Enhanced_Research.ipynb        [Interactive research notebook]
│   └── run_shap_analysis.py                [SHAP interpretability analysis]
│
├── Visualization & Output
│   ├── create_plots.py (493 lines)         [Publication-quality plots]
│   ├── plots/ (generated)                  [Output plots directory]
│   └── model_results.csv (generated)       [Model performance metrics]
│
├── Interactive App
│   └── app_streamlit_diagnostic.py (625 lines)  [Web interface]
│
├── Testing
│   ├── test_diagnostic_system.py           [Diagnostic component tests]
│   ├── test_different_horizons.py          [Horizon sensitivity tests]
│   └── test_subtle_features.py             [Word-level feature tests]
│
├── Documentation
│   ├── README.md                           [Project overview & quick start]
│   ├── RESEARCH_ROADMAP.md                 [Publication timeline & strategy]
│   ├── DIAGNOSTIC_FLOWCHART.md             [System architecture diagrams]
│   ├── STREAMLIT_README.md                 [App documentation]
│   └── requirements_enhanced.txt           [Python dependencies]
│
└── Version Control
    └── .git/                               [Git repository]
```

---

## 7. CURRENT PROJECT STATE & MATURITY

### Development Status: **Advanced / Near-Publication Quality**

### What's Ready
- ✅ **Complete data pipeline**: Load → process → features → market reactions
- ✅ **All NLP approaches implemented**: GPT-4, FinBERT, BART, embeddings
- ✅ **Novel change detection**: 30+ features at sentence and word level
- ✅ **Production models trained**: Random Forest, Gradient Boosting with time-series CV
- ✅ **Interpretability framework**: SHAP analysis for feature importance
- ✅ **Interactive diagnostic system**: Streamlit app with probabilistic forecasting
- ✅ **Test suite**: Validation scripts for all major components
- ✅ **Publication roadmap**: Detailed timeline and paper structure
- ✅ **Documentation**: Comprehensive README, flowcharts, and guides

### What's In Progress
- 🔄 **Holdout set evaluation**: Testing on 2024-2025 data
- 🔄 **Paper writing**: Drafting manuscript sections
- 🔄 **Robustness checks**: Alternative specifications and sensitivity analysis

### What's Planned
- 📋 **Conference submissions**: AFA, WFA, NBER (after paper draft)
- 📋 **Journal submissions**: Journal of Finance, JFE, RFS (top-tier targets)
- 📋 **Extensions**: Intraday analysis, press conference analysis, other central banks

---

## 8. KEY STRENGTHS & INNOVATIONS

### Methodological Innovations
1. **Change Detection as Primary Signal**: Most Fed research focuses on levels; this focuses on changes
2. **Word-Level Linguistic Analysis**: Captures subtle Fed language shifts (transitory→persistent)
3. **Diagnostic Rather Than Point Predictions**: Acknowledges uncertainty with percentile scores and probabilistic forecasts
4. **Multi-Modal NLP Fusion**: Combines GPT-4, FinBERT, BART for robust feature extraction
5. **Proper Time-Series Cross-Validation**: Avoids look-ahead bias with holdout testing

### Publication-Ready Elements
- Novel methodology with strong intuition
- Large dataset (432 statements over 25 years)
- Multiple model validation and robustness checks
- SHAP-based interpretability for reviewer satisfaction
- Clear writing and documentation
- Detailed research roadmap

---

## 9. DEPENDENCIES & REQUIREMENTS

### Core Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **scikit-learn**: ML models and preprocessing
- **shap**: Interpretability analysis
- **nltk**: Text tokenization
- **pandas-datareader**: FRED API access
- **transformers**: GPT-4, FinBERT, BART models
- **torch**: Deep learning backend
- **streamlit**: Web app framework
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plots

See `requirements_enhanced.txt` for full dependencies and versions.

---

## 10. OPPORTUNITIES FOR IMPROVEMENT

### High Priority
1. **Holdout Set Results**: Complete evaluation on 2024-2025 data
2. **Robustness Checks**: Test on 5Y, 10Y yields; different time periods
3. **Economic Significance**: Show trading strategy profitability
4. **Case Studies**: 2-3 detailed examples (March 2023 banking stress, Dec 2021 "transitory" removal)

### Medium Priority
1. **Intraday Data**: If obtainable, test on 1-hour and 30-minute changes
2. **Feature Engineering**: Explore interaction terms, non-linear transformations
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Cross-Central Bank**: Test on ECB, BOE, BOJ communications

### Advanced Extensions
1. **Neural Networks**: LSTM for sequential patterns
2. **Attention Mechanisms**: Which sentences matter most in statements?
3. **Real-Time System**: Deploy for live FOMC analysis
4. **Policy Impact**: Do Fed communications actually affect their own projections?

---

## Conclusion

This is a **sophisticated, well-executed research project** at an advanced stage of development. The combination of multi-modal NLP with novel change detection features is genuinely novel and publishable. The project has strong technical execution, comprehensive documentation, and a clear path to publication in top-tier venues.

The main work remaining is:
1. Completing holdout set evaluation
2. Strengthening the paper narrative
3. Adding robustness checks
4. Submitting to conferences/journals

**Estimated timeline to publication**: 6-18 months (consistent with academic norms)
