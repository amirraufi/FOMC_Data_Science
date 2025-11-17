"""
Test FOMC Market Reactions at Different Time Horizons

This script tests if FOMC communications have effects beyond 1-2 days.

Tests horizons:
- 1 day (immediate)
- 2 days (short-term)
- 5 days (1 week)
- 10 days (2 weeks)
- 20 days (1 month)

This helps answer:
- Do effects persist or reverse?
- What's the optimal prediction horizon?
- Are there delayed reactions?

Usage:
    python test_different_horizons.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pandas_datareader.data as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("TESTING FOMC EFFECTS AT DIFFERENT TIME HORIZONS")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv('data_with_gpt_bart_finbert.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Fetch market data
print("Fetching Treasury yield data from FRED...")
try:
    start_date = df['Date'].min()
    end_date = df['Date'].max() + timedelta(days=30)

    treasury_data = web.DataReader(['DGS2', 'DGS5', 'DGS10'], 'fred', start_date, end_date)
    print(f"✓ Fetched {len(treasury_data)} observations")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Using existing data from CSV...")
    treasury_data = None

# Test different horizons
horizons = [1, 2, 5, 10, 20]
print(f"\nTesting horizons: {horizons} days\n")

results_by_horizon = []

for horizon in horizons:
    print(f"\n{'='*70}")
    print(f"HORIZON: {horizon} DAYS")
    print(f"{'='*70}")

    # Calculate yield changes for this horizon
    target_col = f'dy2_{horizon}d_bp'

    if treasury_data is not None:
        # Calculate from fresh data
        df[target_col] = np.nan

        for idx, row in df.iterrows():
            release_date = row['Release Date']

            # Pre-release
            pre_dates = treasury_data.index[treasury_data.index < release_date]
            if len(pre_dates) == 0:
                continue
            pre_date = pre_dates[-1]

            # Post-release
            target_date = release_date + timedelta(days=horizon)
            post_dates = treasury_data.index[
                (treasury_data.index >= release_date) &
                (treasury_data.index <= target_date + timedelta(days=7))
            ]

            if len(post_dates) < horizon:
                continue

            post_date = post_dates[min(horizon-1, len(post_dates)-1)]

            # Calculate change
            pre_val = treasury_data.loc[pre_date, 'DGS2']
            post_val = treasury_data.loc[post_date, 'DGS2']

            if pd.notna(pre_val) and pd.notna(post_val):
                change_bp = (post_val - pre_val) * 100
                df.loc[idx, target_col] = change_bp
    else:
        # Use existing columns if available
        if horizon == 1 and 'dy2_1d_bp' in df.columns:
            target_col = 'dy2_1d_bp'
        elif horizon == 2 and 'dy2_2d_bp' in df.columns:
            target_col = 'dy2_2d_bp'
        else:
            print(f"  ⚠ No data for {horizon}-day horizon, skipping...")
            continue

    # Check if we have enough data
    valid_data = df[target_col].notna().sum()
    print(f"  Valid observations: {valid_data}")

    if valid_data < 50:
        print(f"  ⚠ Not enough data ({valid_data} < 50), skipping...")
        continue

    # Prepare features
    feature_cols = [col for col in df.columns if (
        col.startswith('gpt_') or
        col.startswith('bart_') or
        col.startswith('finbert_') or
        col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
    )]
    feature_cols = [col for col in feature_cols if col != 'gpt_reason' and col != 'bart_label']
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Extract X and y
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Filter valid samples
    valid_idx = y.notna()
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    # Time-series split
    df_valid = df[valid_idx].copy()
    df_valid['year'] = df_valid['Date'].dt.year

    train_idx = df_valid['year'] < 2017
    test_idx = df_valid['year'] >= 2017

    X_train = X_valid[train_idx.values]
    y_train = y_valid[train_idx.values]
    X_test = X_valid[test_idx.values]
    y_test = y_valid[test_idx.values]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # Directional accuracy
    test_dir_acc = (np.sign(y_pred_test) == np.sign(y_test)).mean()

    # Statistics
    mean_change = y_valid.mean()
    std_change = y_valid.std()

    print(f"\n  Results:")
    print(f"    Mean yield change: {mean_change:.2f} bp")
    print(f"    Std yield change: {std_change:.2f} bp")
    print(f"    Train RMSE: {train_rmse:.2f} bp")
    print(f"    Test RMSE: {test_rmse:.2f} bp")
    print(f"    Test MAE: {test_mae:.2f} bp")
    print(f"    Test R²: {test_r2:.3f}")
    print(f"    Directional Accuracy: {test_dir_acc:.1%}")

    results_by_horizon.append({
        'horizon': horizon,
        'n_samples': valid_data,
        'mean_change': mean_change,
        'std_change': std_change,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'dir_accuracy': test_dir_acc
    })

# Create summary
print("\n" + "="*70)
print("SUMMARY: RESULTS ACROSS ALL HORIZONS")
print("="*70)

results_df = pd.DataFrame(results_by_horizon)
print("\n", results_df.to_string(index=False))

# Save results
results_df.to_csv('horizon_comparison_results.csv', index=False)
print("\n✓ Saved to horizon_comparison_results.csv")

# Create visualization
if len(results_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Across Different Time Horizons', fontsize=14, fontweight='bold')

    # RMSE
    ax1 = axes[0, 0]
    ax1.plot(results_df['horizon'], results_df['test_rmse'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Horizon (days)')
    ax1.set_ylabel('Test RMSE (bp)')
    ax1.set_title('Prediction Error by Horizon')
    ax1.grid(True, alpha=0.3)

    # Directional Accuracy
    ax2 = axes[0, 1]
    ax2.plot(results_df['horizon'], results_df['dir_accuracy'] * 100, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(50, color='red', linestyle='--', label='Random (50%)')
    ax2.set_xlabel('Horizon (days)')
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('Directional Accuracy by Horizon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # R²
    ax3 = axes[1, 0]
    ax3.plot(results_df['horizon'], results_df['test_r2'], 'o-', linewidth=2, markersize=8, color='coral')
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_xlabel('Horizon (days)')
    ax3.set_ylabel('R²')
    ax3.set_title('R² by Horizon')
    ax3.grid(True, alpha=0.3)

    # Market volatility
    ax4 = axes[1, 1]
    ax4.plot(results_df['horizon'], results_df['std_change'], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Horizon (days)')
    ax4.set_ylabel('Std Dev of Yield Change (bp)')
    ax4.set_title('Market Volatility by Horizon')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/horizon_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot to plots/horizon_comparison.png")
    plt.close()

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("""
Key Questions Answered:

1. Do effects persist or reverse?
   - Compare mean_change across horizons
   - If signs flip, effects reverse
   - If magnitude decreases, effects fade

2. What's the optimal prediction horizon?
   - Look for highest directional accuracy
   - Balance between predictability and relevance

3. Are there delayed reactions?
   - Check if 5-day or 10-day effects are larger
   - May indicate slow information absorption

NEXT STEPS:
- Use horizon with best directional accuracy for paper
- Mention other horizons as robustness checks
- If 1-day is best, market reacts immediately (efficient)
- If 5-10 day is best, delayed reaction story
""")

print(f"\nCompleted: {pd.Timestamp.now()}")
print("="*70)
