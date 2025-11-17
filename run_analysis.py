"""
FOMC Market Reaction Analysis - Complete Pipeline

This script does 5 things:
1. Loads data_with_gpt_bart_finbert.csv
2. Adds change detection features
3. Merges with fresh FRED data (DFF)
4. Trains models with time-series CV
5. Runs SHAP analysis

After Step 3, saves enhanced dataset to: data_enhanced_with_changes.csv
This file includes all NLP features + change detection + DFF reactions.

Usage:
    python run_analysis.py

To skip Steps 1-3 next time:
    - Load data_enhanced_with_changes.csv directly
    - Jump to Step 4 (model training)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# SHAP for interpretability
import shap

# Data fetching
import pandas_datareader.data as web

# Text processing for change detection
import nltk
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher

# Import word-level linguistic analyzer (NEW!)
from fomc_analysis_utils import SubtleLinguisticAnalyzer

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


print("="*70)
print("FOMC MARKET REACTION ANALYSIS - COMPLETE PIPELINE")
print("="*70)
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# Load existing features - try multiple sources
print("\nAttempting to load data...")
try:
    # Try original NLP-enhanced data first
    print("  Trying data_with_gpt_bart_finbert.csv...")
    df = pd.read_csv('data_with_gpt_bart_finbert.csv')
    print("  ✓ Loaded base NLP data")
except FileNotFoundError:
    # Fallback to enhanced data (already has sentence-level changes)
    print("  Not found. Trying data_enhanced_with_changes.csv...")
    df = pd.read_csv('data_enhanced_with_changes.csv')
    print("  ✓ Loaded enhanced data (will add word-level features)")

# Parse dates
df['Date'] = pd.to_datetime(df['Date'])
if 'Release Date' in df.columns:
    df['Release Date'] = pd.to_datetime(df['Release Date'])

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

print(f"✓ Loaded {len(df)} FOMC statements")
print(f"  Date range: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
print(f"  Columns: {len(df.columns)}")


# ============================================================================
# STEP 2: ADD CHANGE DETECTION FEATURES
# ============================================================================
print("\n" + "="*70)
print("STEP 2: ADDING CHANGE DETECTION FEATURES")
print("="*70)

def compute_text_similarity(text1, text2):
    """Compute similarity between two texts"""
    if pd.isna(text1) or pd.isna(text2):
        return np.nan
    return SequenceMatcher(None, text1, text2).ratio()


def extract_key_phrases(text):
    """Extract key policy-related phrases"""
    if pd.isna(text):
        return {}

    text_lower = text.lower()

    phrases = {
        # Inflation language
        'inflation_elevated': 'inflation remains elevated' in text_lower or 'elevated inflation' in text_lower,
        'inflation_moderating': 'inflation has moderated' in text_lower or 'moderating inflation' in text_lower,
        'inflation_easing': 'inflation easing' in text_lower or 'inflation has eased' in text_lower,

        # Rate language
        'rate_increases': 'rate increase' in text_lower or 'raising the target range' in text_lower,
        'rate_cuts': 'rate cut' in text_lower or 'lowering the target range' in text_lower,
        'rate_hold': 'maintain the target range' in text_lower,

        # Forward guidance
        'data_dependent': 'data dependent' in text_lower or 'incoming data' in text_lower,
        'patient': 'patient' in text_lower and 'policy' in text_lower,

        # Labor market
        'labor_tight': 'tight labor' in text_lower or 'labor market remains tight' in text_lower,
        'labor_softening': 'labor market has softened' in text_lower,

        # Economic outlook
        'growth_solid': 'solid growth' in text_lower,
        'growth_slowing': 'slowing growth' in text_lower,
    }

    return phrases


def detect_statement_changes(current_text, previous_text):
    """Detect changes between consecutive FOMC statements"""
    if pd.isna(current_text) or pd.isna(previous_text):
        return {}

    # Tokenize into sentences
    try:
        curr_sentences = sent_tokenize(current_text)
        prev_sentences = sent_tokenize(previous_text)
    except:
        # Fallback if tokenization fails
        curr_sentences = current_text.split('.')
        prev_sentences = previous_text.split('.')

    # Convert to sets for comparison
    curr_set = set(s.strip() for s in curr_sentences if s.strip())
    prev_set = set(s.strip() for s in prev_sentences if s.strip())

    # Count changes
    added = curr_set - prev_set
    removed = prev_set - curr_set
    unchanged = curr_set & prev_set

    # Overall similarity
    overall_similarity = compute_text_similarity(current_text, previous_text)

    # Length changes
    len_change_pct = (len(current_text) - len(previous_text)) / len(previous_text) * 100 if len(previous_text) > 0 else 0
    sentence_count_change = len(curr_sentences) - len(prev_sentences)

    # Key phrase analysis
    curr_phrases = extract_key_phrases(current_text)
    prev_phrases = extract_key_phrases(previous_text)

    # Track phrase changes
    phrase_changes = {}
    for phrase_name in curr_phrases.keys():
        curr_val = curr_phrases[phrase_name]
        prev_val = prev_phrases[phrase_name]

        if curr_val and not prev_val:
            phrase_changes[f'change_{phrase_name}_added'] = 1
        elif not curr_val and prev_val:
            phrase_changes[f'change_{phrase_name}_removed'] = 1
        else:
            phrase_changes[f'change_{phrase_name}_added'] = 0
            phrase_changes[f'change_{phrase_name}_removed'] = 0

    # Compile features
    features = {
        'change_sentences_added': len(added),
        'change_sentences_removed': len(removed),
        'change_sentences_unchanged': len(unchanged),
        'change_net_sentences': len(added) - len(removed),
        'change_pct_sentences_modified': (len(added) + len(removed)) / max(len(prev_set), 1) * 100,
        'change_overall_similarity': overall_similarity,
        'change_text_length_pct': len_change_pct,
        'change_sentence_count': sentence_count_change,
    }

    # Add phrase changes
    features.update(phrase_changes)

    return features


# Load communications.csv to get the text
print("\nLoading communications.csv for text data...")
try:
    comm_df = pd.read_csv('communications.csv')
    comm_df['Date'] = pd.to_datetime(comm_df['Date'])

    # Filter to ONLY Statements (not Minutes)
    comm_df = comm_df[comm_df['Type'] == 'Statement'].copy()
    print(f"  Filtered to {len(comm_df)} Statements (excluded Minutes)")

    # Merge to get text
    df = df.merge(comm_df[['Date', 'Text']], on='Date', how='left')
    print(f"✓ Merged with communications data")
except Exception as e:
    print(f"✗ Error loading communications.csv: {e}")
    print("  Continuing without change detection features...")
    df['Text'] = None

# Add change features
if 'Text' in df.columns and df['Text'].notna().sum() > 0:
    print("\nComputing change detection features...")
    print("  - Sentence-level changes (existing)")
    print("  - Word-level linguistic features (NEW!)")

    all_change_features = []

    for idx in range(len(df)):
        if idx == 0 or pd.isna(df.loc[idx, 'Text']):
            # First statement or missing text
            all_change_features.append({})
        else:
            current_text = df.loc[idx, 'Text']
            previous_text = df.loc[idx-1, 'Text']

            # Sentence-level changes (existing)
            changes = detect_statement_changes(current_text, previous_text)

            # Word-level linguistic features (NEW!)
            subtle_features = SubtleLinguisticAnalyzer.analyze_all(current_text, previous_text)

            # Combine both
            changes.update(subtle_features)
            all_change_features.append(changes)

    # Convert to DataFrame and concatenate
    change_df = pd.DataFrame(all_change_features)
    df = pd.concat([df, change_df], axis=1)

    print(f"✓ Added {len(change_df.columns)} change detection features")
    print(f"  (Includes ~32 sentence-level + ~20 word-level features)")
else:
    print("⚠ No text data available, skipping change detection")


# ============================================================================
# STEP 3: MERGE WITH FRESH FRED DATA (DFF)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: FETCHING FRESH FRED DATA")
print("="*70)

print("\nFetching Effective Federal Funds Rate (DFF) from FRED...")
try:
    # Fetch DFF data
    start_date = df['Date'].min()
    end_date = df['Date'].max() + timedelta(days=10)

    dff_data = web.DataReader('DFF', 'fred', start_date, end_date)
    print(f"✓ Fetched {len(dff_data)} DFF observations")

    # Calculate DFF changes for each FOMC event
    print("\nCalculating DFF reactions...")

    for horizon in [1, 2]:
        df[f'dff_{horizon}d_bp'] = np.nan

        for idx, row in df.iterrows():
            release_date = row['Release Date']

            # Get pre-release value
            pre_dates = dff_data.index[dff_data.index < release_date]
            if len(pre_dates) == 0:
                continue
            pre_date = pre_dates[-1]

            # Get post-release value
            target_date = release_date + timedelta(days=horizon)
            post_dates = dff_data.index[
                (dff_data.index >= release_date) &
                (dff_data.index <= target_date + timedelta(days=5))
            ]

            if len(post_dates) == 0:
                continue

            post_date = post_dates[min(horizon-1, len(post_dates)-1)]

            # Compute change in basis points
            pre_val = dff_data.loc[pre_date, 'DFF']
            post_val = dff_data.loc[post_date, 'DFF']

            if pd.notna(pre_val) and pd.notna(post_val):
                change_bp = (post_val - pre_val) * 100
                df.loc[idx, f'dff_{horizon}d_bp'] = change_bp

    print(f"✓ Added DFF reaction features (dff_1d_bp, dff_2d_bp)")

except Exception as e:
    print(f"✗ Error fetching FRED data: {e}")
    print("  Continuing without DFF features...")

# Save enhanced dataset
print("\nSaving enhanced dataset with all features...")
# Drop the Text column to save space (it's large and not needed for modeling)
df_to_save = df.drop(columns=['Text'], errors='ignore')
df_to_save.to_csv('data_enhanced_with_changes.csv', index=False)
print(f"✓ Saved to 'data_enhanced_with_changes.csv'")
print(f"  Shape: {df_to_save.shape[0]} rows x {df_to_save.shape[1]} columns")
print(f"  Includes: NLP features + change detection + DFF reactions")


# ============================================================================
# STEP 4: TRAIN MODELS WITH TIME-SERIES CV
# ============================================================================
print("\n" + "="*70)
print("STEP 4: TRAINING MODELS WITH TIME-SERIES CV")
print("="*70)

# Prepare feature matrix
print("\nPreparing feature matrix...")

# Select features
feature_cols = [col for col in df.columns if (
    col.startswith('change_') or
    col.startswith('subtle_') or  # NEW: Word-level linguistic features!
    col.startswith('gpt_') or
    col.startswith('bart_') or
    col.startswith('finbert_') or
    col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
)]

# Remove non-numeric columns
feature_cols = [col for col in feature_cols if col != 'gpt_reason' and col != 'bart_label']

# Filter to available columns
feature_cols = [col for col in feature_cols if col in df.columns]

print(f"  Features selected: {len(feature_cols)}")
print(f"  Sample features: {feature_cols[:5]}...")

# Target variable
target = 'dy2_1d_bp'  # 2-year Treasury yield 1-day change

# Extract X and y
X = df[feature_cols].copy()
y = df[target].copy()

# Handle missing values
X = X.fillna(0)

# Filter to valid samples
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]
dates = df.loc[valid_idx, 'Date'].values

print(f"\n  Final dataset: {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Target: {target}")
print(f"  Target stats: mean={y.mean():.2f} bp, std={y.std():.2f} bp")

# Create train/validation/holdout splits
print("\nCreating time-series splits...")
df_valid = df[valid_idx].copy()
df_valid['year'] = pd.to_datetime(df_valid['Date']).dt.year

train_idx = df_valid['year'] < 2017
val_idx = (df_valid['year'] >= 2017) & (df_valid['year'] < 2024)
holdout_idx = df_valid['year'] >= 2024

X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]
X_holdout = X[holdout_idx]
y_holdout = y[holdout_idx]

print(f"  Train: {len(X_train)} samples (pre-2017)")
print(f"  Validation: {len(X_val)} samples (2017-2023)")
print(f"  Holdout: {len(X_holdout)} samples (2024+)")

# Time-series cross-validation
print("\nRunning time-series cross-validation...")

tscv = TimeSeriesSplit(n_splits=5)
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}

results = []

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")

    cv_scores = []

    for fold_idx, (train_fold_idx, test_fold_idx) in enumerate(tscv.split(X_trainval)):
        X_train_fold = X_trainval.iloc[train_fold_idx]
        y_train_fold = y_trainval.iloc[train_fold_idx]
        X_test_fold = X_trainval.iloc[test_fold_idx]
        y_test_fold = y_trainval.iloc[test_fold_idx]

        # Train
        model.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred = model.predict(X_test_fold)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        mae = mean_absolute_error(y_test_fold, y_pred)
        r2 = r2_score(y_test_fold, y_pred)

        cv_scores.append({'rmse': rmse, 'mae': mae, 'r2': r2})

    # Average CV scores
    cv_df = pd.DataFrame(cv_scores)
    mean_rmse = cv_df['rmse'].mean()
    mean_mae = cv_df['mae'].mean()
    mean_r2 = cv_df['r2'].mean()

    # Train on full train+val set and evaluate on holdout
    model.fit(X_trainval, y_trainval)

    if len(X_holdout) > 0:
        y_holdout_pred = model.predict(X_holdout)
        holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_holdout_pred))
        holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
        holdout_r2 = r2_score(y_holdout, y_holdout_pred)
    else:
        holdout_rmse = holdout_mae = holdout_r2 = np.nan

    results.append({
        'model': model_name,
        'cv_rmse': mean_rmse,
        'cv_mae': mean_mae,
        'cv_r2': mean_r2,
        'holdout_rmse': holdout_rmse,
        'holdout_mae': holdout_mae,
        'holdout_r2': holdout_r2,
    })

    print(f"    CV RMSE: {mean_rmse:.2f} bp")
    print(f"    Holdout RMSE: {holdout_rmse:.2f} bp" if not np.isnan(holdout_rmse) else "    Holdout: N/A")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print(f"\n✓ Saved model comparison to model_results.csv")

# Select best model
best_model_name = results_df.loc[results_df['cv_rmse'].idxmin(), 'model']
best_model = models[best_model_name]
print(f"\n✓ Best model: {best_model_name}")


# ============================================================================
# STEP 5: RUN SHAP ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: RUNNING SHAP ANALYSIS")
print("="*70)

print(f"\nComputing SHAP values for {best_model_name}...")

# Train best model on train+val
best_model.fit(X_trainval, y_trainval)

# Compute SHAP values
try:
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_val)
    else:
        # For linear models
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_val)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    # Save
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"✓ Saved feature importance to feature_importance.csv")

    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    # Generate SHAP summary plot
    print(f"\nGenerating SHAP summary plot...")
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved SHAP plot to shap_summary_plot.png")

except Exception as e:
    print(f"✗ Error computing SHAP values: {e}")
    print(f"  Continuing without SHAP analysis...")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

print(f"\n✓ Output files created:")
print(f"  - model_results.csv (model comparison)")
print(f"  - feature_importance.csv (SHAP-based feature rankings)")
print(f"  - shap_summary_plot.png (feature importance visualization)")

print(f"\n✓ Key results:")
print(f"  - Dataset: {len(df)} FOMC statements ({df['Date'].min():%Y} - {df['Date'].max():%Y})")
print(f"  - Features: {len(feature_cols)} total")
print(f"  - Best model: {best_model_name}")
print(f"  - CV RMSE: {results_df.loc[results_df['model'] == best_model_name, 'cv_rmse'].values[0]:.2f} bp")

print(f"\n✓ Next steps:")
print(f"  1. Review model_results.csv for all model comparisons")
print(f"  2. Check feature_importance.csv to see which features matter most")
print(f"  3. Examine shap_summary_plot.png for visual feature importance")
print(f"  4. Use these results for your paper!")

print(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("="*70)
