"""
Re-run SHAP analysis on the best model with word-level features
Fixes the array length mismatch issue
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

print("=" * 70)
print("SHAP ANALYSIS WITH WORD-LEVEL FEATURES")
print("=" * 70)

# Load enhanced data
print("\nLoading data_enhanced_with_changes.csv...")
df = pd.read_csv('data_enhanced_with_changes.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"✓ Loaded {len(df)} statements with {len(df.columns)} columns")

# Select features (same as run_analysis.py)
feature_cols = [col for col in df.columns if (
    col.startswith('change_') or
    col.startswith('subtle_') or  # Word-level features
    col.startswith('gpt_') or
    col.startswith('bart_') or
    col.startswith('finbert_') or
    col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
)]

# Remove non-numeric columns
feature_cols = [col for col in feature_cols if col != 'gpt_reason' and col != 'bart_label']
feature_cols = [col for col in feature_cols if col in df.columns]

print(f"  Features: {len(feature_cols)}")
print(f"  Word-level features: {len([c for c in feature_cols if c.startswith('subtle_')])}")

# Target
target = 'dy2_1d_bp'

# Extract X and y
X = df[feature_cols].fillna(0)
y = df[target]

# Filter valid samples
valid_idx = y.notna()
X = X[valid_idx].copy()
y = y[valid_idx].copy()
df_valid = df[valid_idx].copy()

print(f"\n  Valid samples: {len(X)} x {X.shape[1]} features")

# Create splits
df_valid['year'] = df_valid['Date'].dt.year
train_idx = df_valid['year'] < 2017
val_idx = (df_valid['year'] >= 2017) & (df_valid['year'] < 2024)

X_train = X[train_idx].values
y_train = y[train_idx].values
X_val = X[val_idx].values
y_val = y[val_idx].values
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

# Train best model (Random Forest from previous results)
print("\nTraining Random Forest (best model)...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
model.fit(X_trainval, y_trainval)
print("✓ Model trained")

# Compute SHAP values
print("\nComputing SHAP values...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Feature count: {len(feature_cols)}")

    # Ensure shapes match
    if shap_values.shape[1] != len(feature_cols):
        print(f"✗ Shape mismatch: {shap_values.shape[1]} != {len(feature_cols)}")
        print("  Adjusting feature list...")
        feature_cols = feature_cols[:shap_values.shape[1]]

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    # Save
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"\n✓ Saved feature importance to feature_importance.csv")

    # Show top features
    print(f"\nTop 20 most important features:")
    print(feature_importance.head(20).to_string(index=False))

    # Word-level features in top rankings
    subtle_features = feature_importance[feature_importance['feature'].str.startswith('subtle_')]
    if len(subtle_features) > 0:
        print(f"\n✓ Word-level features in rankings:")
        print(f"  Highest ranked: {subtle_features.iloc[0]['feature']} (rank #{feature_importance.index[feature_importance['feature'] == subtle_features.iloc[0]['feature']].tolist()[0] + 1})")
        print(f"  Total ranked: {len(subtle_features)}")
        print("\nTop 5 word-level features:")
        print(subtle_features.head(5).to_string(index=False))
    else:
        print("\n⚠ No word-level features in top rankings")

    # Generate SHAP summary plot
    print(f"\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to shap_summary_plot.png")

except Exception as e:
    print(f"✗ Error computing SHAP values: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ SHAP ANALYSIS COMPLETE!")
print("=" * 70)
