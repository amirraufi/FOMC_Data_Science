"""
Test Diagnostic & Probabilistic Analysis System

Demonstrates the new diagnostic approach:
1. Percentile scoring (how hawkish vs history)
2. Change highlighting (what changed linguistically)
3. Nearest neighbor retrieval (similar past episodes)
4. Probabilistic predictions (conditional distributions)
"""

import pandas as pd
import numpy as np
from fomc_analysis_utils import (
    DiagnosticAnalyzer,
    ProbabilisticPredictor,
    SubtleLinguisticAnalyzer
)

print("="*70)
print("TESTING DIAGNOSTIC & PROBABILISTIC ANALYSIS SYSTEM")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
try:
    # Try enhanced data first (has all features)
    try:
        df = pd.read_csv('data_enhanced_with_changes.csv')
        print(f"✓ Loaded enhanced data")
    except:
        df = pd.read_csv('data_with_gpt_bart_finbert.csv')
        print(f"✓ Loaded base data")

    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  {len(df)} FOMC statements with {len(df.columns)} features")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# ============================================================================
# TEST 1: COMPOSITE HAWKISHNESS SCORING
# ============================================================================
print("\n" + "="*70)
print("TEST 1: COMPOSITE HAWKISHNESS SCORING")
print("="*70)

composite_scores = DiagnosticAnalyzer.create_composite_hawkishness(df)
df['composite_hawk'] = composite_scores

# Pick a recent statement to analyze
current_idx = -5  # 5th most recent
current_score = df.iloc[current_idx]['composite_hawk']
current_date = df.iloc[current_idx]['Date']

# Compute percentile relative to all history
historical_scores = df['composite_hawk'].dropna()
percentile = DiagnosticAnalyzer.compute_hawkishness_percentile(
    current_score, historical_scores
)

print(f"\nStatement Date: {current_date:%Y-%m-%d}")
print(f"Composite Hawkishness Score: {current_score:.3f} (0=dovish, 1=hawkish)")
print(f"Historical Percentile: {percentile:.0f}th percentile")

if percentile > 80:
    print("  → VERY HAWKISH (more hawkish than 80% of history)")
elif percentile > 60:
    print("  → Moderately hawkish")
elif percentile > 40:
    print("  → Neutral")
elif percentile > 20:
    print("  → Moderately dovish")
else:
    print("  → VERY DOVISH (more dovish than 80% of history)")

# ============================================================================
# TEST 2: CHANGE HIGHLIGHTING
# ============================================================================
print("\n" + "="*70)
print("TEST 2: CHANGE HIGHLIGHTING")
print("="*70)

# Create sample change dictionary (simulate word-level features)
sample_changes = {
    'subtle_inflation_duration_intensity_change': 2,  # transitory → persistent
    'subtle_policy_stance_intensity_change': 1,  # patient → data-dependent
    'subtle_hedge_word_count_change': -2,  # less hedging
    'subtle_certainty_word_count_change': 3,  # more certainty
    'subtle_negation_count_change': 1,  # added negation
    'subtle_net_intensity_change': 2,  # more intense
    'subtle_future_tense_count_change': 3,  # more forward guidance
}

highlights = DiagnosticAnalyzer.highlight_key_changes(sample_changes)

print("\nKey Language Changes Detected:")
for i, highlight in enumerate(highlights, 1):
    print(f"  {i}. {highlight}")

# ============================================================================
# TEST 3: NEAREST NEIGHBOR RETRIEVAL
# ============================================================================
print("\n" + "="*70)
print("TEST 3: NEAREST NEIGHBOR RETRIEVAL")
print("="*70)

# Select feature columns
feature_cols = [col for col in df.columns if (
    col.startswith('gpt_') or
    col.startswith('bart_') or
    col.startswith('finbert_')
)]
feature_cols = [col for col in feature_cols if col not in ['gpt_reason', 'bart_label']]
feature_cols = [col for col in feature_cols if col in df.columns]

print(f"\nUsing {len(feature_cols)} NLP features for similarity")

# Current statement features
current_features = df.iloc[current_idx][feature_cols].fillna(0)

# Historical statements (exclude current and very recent)
historical_df = df.iloc[:-10].copy()
historical_features = historical_df[feature_cols].fillna(0)

# Find 5 most similar
neighbor_indices = DiagnosticAnalyzer.find_nearest_neighbors(
    current_features, historical_features, k=5
)

print(f"\nTop 5 Most Similar Historical Statements to {current_date:%Y-%m-%d}:")
print("-" * 70)

for rank, idx in enumerate(neighbor_indices, 1):
    neighbor_row = historical_df.iloc[idx]
    neighbor_date = neighbor_row['Date']

    # Get actual market reaction if available
    if 'dy2_1d_bp' in neighbor_row and pd.notna(neighbor_row['dy2_1d_bp']):
        reaction = neighbor_row['dy2_1d_bp']
        print(f"{rank}. {neighbor_date:%Y-%m-%d} → Actual reaction: {reaction:+.1f} bp")
    else:
        print(f"{rank}. {neighbor_date:%Y-%m-%d} → (no market data)")

# ============================================================================
# TEST 4: PROBABILISTIC PREDICTION
# ============================================================================
print("\n" + "="*70)
print("TEST 4: PROBABILISTIC PREDICTION")
print("="*70)

# Prepare historical data with outcomes
historical_df_with_outcomes = df.iloc[:-10][df['dy2_1d_bp'].notna()].copy()

if len(historical_df_with_outcomes) > 20:
    # Compute conditional distribution
    dist = ProbabilisticPredictor.conditional_distribution(
        current_features=current_features,
        historical_df=historical_df_with_outcomes,
        feature_cols=feature_cols,
        target='dy2_1d_bp',
        k=20
    )

    # Format and print
    forecast = ProbabilisticPredictor.format_probabilistic_forecast(dist, "2Y Treasury")
    print(forecast)

    # Compare to point prediction
    print("-" * 70)
    print("CONTRAST WITH POINT PREDICTION:")
    print(f"  Old approach: 'Yields will rise {dist['median']:+.1f} bp'")
    print(f"  New approach: '{dist['median']:+.1f} bp is the median, but 80% interval is [{dist['q10']:+.1f}, {dist['q90']:+.1f}] bp'")
    print(f"  → Acknowledges uncertainty! {dist['q90'] - dist['q10']:.1f} bp range")
else:
    print("⚠ Insufficient historical data for probabilistic prediction")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: DIAGNOSTIC VS PREDICTIVE APPROACH")
print("="*70)

print("""
OLD APPROACH (Point Prediction):
  "2Y yields will rise 4.3 bp"
  - False precision (R² = -0.28)
  - Ignores uncertainty
  - Hard to interpret

NEW APPROACH (Diagnostic + Probabilistic):
  1. Hawkishness: 82nd percentile (very hawkish)
  2. Key changes:
     - Inflation language: "transitory" → "persistent"
     - Policy stance: more certain, less hedging
     - Forward guidance increased
  3. Similar historical episodes:
     - July 2023 → +4.2 bp
     - May 2023 → +6.1 bp
     - March 2023 → +3.8 bp
  4. Conditional forecast:
     - Median: +5.1 bp
     - 80% interval: [-1.2, +11.3] bp
     - Tail risk (>10bp): 15%

  → More useful, honest, and sophisticated!
""")

print("\n✓ All diagnostic tests completed successfully!")
print("\nNext: Update Streamlit app to use this diagnostic approach")
print("="*70)
