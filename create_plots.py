"""
FOMC Analysis - Comprehensive Plotting Script

Creates publication-quality plots for:
1. EDA (Exploratory Data Analysis)
2. Model Performance Comparison
3. Feature Importance Visualizations
4. Case Studies

Usage:
    python create_plots.py

Requires:
    - data_enhanced_with_changes.csv (from run_analysis.py)
    - model_results.csv (from run_analysis.py)
    - feature_importance.csv (from run_analysis.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("="*70)
print("FOMC ANALYSIS - CREATING ALL PLOTS")
print("="*70)
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

df = pd.read_csv('data_enhanced_with_changes.csv')
df['Date'] = pd.to_datetime(df['Date'])

try:
    model_results = pd.read_csv('model_results.csv')
    has_models = True
except:
    print("  ⚠ model_results.csv not found - skipping model plots")
    has_models = False

try:
    feature_importance = pd.read_csv('feature_importance.csv')
    has_features = True
except:
    print("  ⚠ feature_importance.csv not found - skipping feature plots")
    has_features = False

print(f"✓ Loaded {len(df)} observations\n")

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# ============================================================================
# SECTION 1: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("="*70)
print("SECTION 1: EXPLORATORY DATA ANALYSIS")
print("="*70)

# --------------------------------------------------------------------------
# Figure 1: Distribution of Market Reactions
# --------------------------------------------------------------------------
print("\n1. Creating market reaction distributions...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Treasury Yield Reactions to FOMC Communications',
             fontsize=14, fontweight='bold', y=0.995)

targets = ['dy2_1d_bp', 'dy5_1d_bp', 'dy10_1d_bp',
           'dy2_2d_bp', 'dy5_2d_bp', 'dy10_2d_bp']
titles = ['2Y Treasury (1-day)', '5Y Treasury (1-day)', '10Y Treasury (1-day)',
          '2Y Treasury (2-day)', '5Y Treasury (2-day)', '10Y Treasury (2-day)']

for idx, (target, title) in enumerate(zip(targets, titles)):
    ax = axes[idx // 3, idx % 3]

    data = df[target].dropna()

    # Histogram with KDE
    ax.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)

    # Add KDE line
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Statistics
    mean_val = data.mean()
    std_val = data.std()
    ax.text(0.05, 0.95, f'Mean: {mean_val:.2f} bp\nStd: {std_val:.2f} bp\nN: {len(data)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Yield Change (basis points)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('plots/figure1_market_reactions_distribution.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/figure1_market_reactions_distribution.png")

# --------------------------------------------------------------------------
# Figure 2: Time Series of Market Reactions
# --------------------------------------------------------------------------
print("2. Creating market reaction time series...")

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle('Treasury Yield Reactions Over Time', fontsize=14, fontweight='bold')

maturities = [('dy2_1d_bp', '2-Year'), ('dy5_1d_bp', '5-Year'), ('dy10_1d_bp', '10-Year')]

for idx, (target, label) in enumerate(maturities):
    ax = axes[idx]

    data = df[['Date', target]].dropna()

    # Scatter plot with color coding
    colors = ['red' if x > 0 else 'blue' for x in data[target]]
    ax.scatter(data['Date'], data[target], c=colors, alpha=0.6, s=50)

    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Rolling mean (6-month)
    data_sorted = data.sort_values('Date')
    rolling_mean = data_sorted[target].rolling(window=6, center=True).mean()
    ax.plot(data_sorted['Date'], rolling_mean, 'g-', linewidth=2, label='6-month moving average')

    ax.set_ylabel(f'{label} Yield Change (bp)', fontsize=10)
    ax.set_title(f'{label} Treasury Yield 1-Day Reaction', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Date', fontsize=10)
plt.tight_layout()
plt.savefig('plots/figure2_market_reactions_timeseries.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/figure2_market_reactions_timeseries.png")

# --------------------------------------------------------------------------
# Figure 3: NLP Features Distribution
# --------------------------------------------------------------------------
print("3. Creating NLP features distribution...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of NLP Features', fontsize=14, fontweight='bold', y=0.995)

nlp_features = [
    ('gpt_hawk_score', 'GPT-4 Hawkishness Score'),
    ('bart_score', 'BART Sentiment Score'),
    ('finbert_score', 'FinBERT Sentiment Score'),
    ('hawk_minus_dove', 'Hawk - Dove Word Count'),
    ('delta_semantic', 'Semantic Change'),
    ('cos_prev', 'Cosine Similarity to Previous')
]

for idx, (feature, title) in enumerate(nlp_features):
    ax = axes[idx // 3, idx % 3]

    if feature in df.columns:
        data = df[feature].dropna()

        ax.hist(data, bins=25, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')

        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f'{feature}\nNot Available',
                ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('plots/figure3_nlp_features_distribution.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/figure3_nlp_features_distribution.png")

# --------------------------------------------------------------------------
# Figure 4: Feature Correlation Heatmap
# --------------------------------------------------------------------------
print("4. Creating correlation heatmap...")

# Select key features for correlation
key_features = ['gpt_hawk_score', 'bart_score', 'finbert_score', 'hawk_minus_dove',
                'delta_semantic', 'cos_prev', 'dy2_1d_bp', 'dy5_1d_bp', 'dy10_1d_bp']

available_features = [f for f in key_features if f in df.columns]
corr_df = df[available_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Matrix: NLP Features and Market Reactions',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plots/figure4_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/figure4_correlation_heatmap.png")

# --------------------------------------------------------------------------
# Figure 5: Change Detection Features
# --------------------------------------------------------------------------
print("5. Creating change detection visualization...")

change_features = [col for col in df.columns if col.startswith('change_')][:8]

if len(change_features) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Distribution of Change Detection Features', fontsize=14, fontweight='bold')

    for idx, feature in enumerate(change_features):
        ax = axes[idx // 4, idx % 4]

        data = df[feature].dropna()
        ax.hist(data, bins=20, alpha=0.7, edgecolor='black', color='coral')
        ax.set_title(feature.replace('change_', '').replace('_', ' ').title(),
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/figure5_change_detection_features.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/figure5_change_detection_features.png")

# ============================================================================
# SECTION 2: MODEL PERFORMANCE COMPARISON
# ============================================================================
if has_models:
    print("\n" + "="*70)
    print("SECTION 2: MODEL PERFORMANCE COMPARISON")
    print("="*70)

    # --------------------------------------------------------------------------
    # Figure 6: Model Comparison - RMSE
    # --------------------------------------------------------------------------
    print("\n6. Creating model comparison (RMSE)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

    # CV RMSE
    models = model_results['model']
    cv_rmse = model_results['cv_rmse']

    bars1 = ax1.bar(models, cv_rmse, color=['steelblue', 'coral', 'lightgreen', 'gold'])
    ax1.set_ylabel('RMSE (basis points)', fontsize=11)
    ax1.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Holdout RMSE
    holdout_rmse = model_results['holdout_rmse']
    bars2 = ax2.bar(models, holdout_rmse, color=['steelblue', 'coral', 'lightgreen', 'gold'])
    ax2.set_ylabel('RMSE (basis points)', fontsize=11)
    ax2.set_title('Holdout Test Performance (2024-2025)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/figure6_model_comparison_rmse.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/figure6_model_comparison_rmse.png")

    # --------------------------------------------------------------------------
    # Figure 7: Model Comparison - All Metrics
    # --------------------------------------------------------------------------
    print("7. Creating comprehensive model comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance: All Metrics', fontsize=14, fontweight='bold')

    metrics = [
        ('cv_rmse', 'RMSE (lower is better)'),
        ('cv_mae', 'MAE (lower is better)'),
        ('cv_r2', 'R² (higher is better)')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        values = model_results[metric]
        bars = ax.bar(models, values, color=['steelblue', 'coral', 'lightgreen', 'gold'])
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(metric.upper().replace('_', ' '), fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add values
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/figure7_model_comparison_all_metrics.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/figure7_model_comparison_all_metrics.png")

# ============================================================================
# SECTION 3: FEATURE IMPORTANCE
# ============================================================================
if has_features:
    print("\n" + "="*70)
    print("SECTION 3: FEATURE IMPORTANCE")
    print("="*70)

    # --------------------------------------------------------------------------
    # Figure 8: Top 20 Features
    # --------------------------------------------------------------------------
    print("\n8. Creating feature importance plot...")

    top_features = feature_importance.head(20)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.8)

    # Highlight change detection features
    for idx, feature in enumerate(top_features['feature']):
        if 'change_' in feature:
            bars[idx].set_color('coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
    ax.invert_yaxis()
    ax.set_xlabel('SHAP Importance (mean |SHAP value|)', fontsize=11)
    ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='NLP Features'),
        Patch(facecolor='coral', label='Change Detection Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('plots/figure8_feature_importance_top20.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/figure8_feature_importance_top20.png")

    # --------------------------------------------------------------------------
    # Figure 9: Feature Groups Comparison
    # --------------------------------------------------------------------------
    print("9. Creating feature groups comparison...")

    # Categorize features
    feature_importance['category'] = 'Other'
    feature_importance.loc[feature_importance['feature'].str.contains('change_'), 'category'] = 'Change Detection'
    feature_importance.loc[feature_importance['feature'].str.contains('gpt_'), 'category'] = 'GPT-4'
    feature_importance.loc[feature_importance['feature'].str.contains('bart_'), 'category'] = 'BART'
    feature_importance.loc[feature_importance['feature'].str.contains('finbert_'), 'category'] = 'FinBERT'
    feature_importance.loc[feature_importance['feature'].str.contains('hawk|dove'), 'category'] = 'Keyword Count'

    category_importance = feature_importance.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
    category_importance = category_importance.sort_values('sum', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Importance by Category', fontsize=14, fontweight='bold')

    # Total importance
    ax1.bar(category_importance.index, category_importance['sum'],
            color=['coral', 'steelblue', 'lightgreen', 'gold', 'purple'][:len(category_importance)])
    ax1.set_ylabel('Total SHAP Importance', fontsize=11)
    ax1.set_title('Total Importance by Category', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Average importance
    ax2.bar(category_importance.index, category_importance['mean'],
            color=['coral', 'steelblue', 'lightgreen', 'gold', 'purple'][:len(category_importance)])
    ax2.set_ylabel('Average SHAP Importance', fontsize=11)
    ax2.set_title('Average Importance by Category', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('plots/figure9_feature_groups_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/figure9_feature_groups_comparison.png")

# ============================================================================
# SECTION 4: SCATTER PLOTS - NLP vs Market Reactions
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: NLP FEATURES VS MARKET REACTIONS")
print("="*70)

print("\n10. Creating scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('NLP Features vs 2Y Treasury Yield Reaction', fontsize=14, fontweight='bold', y=0.995)

nlp_vars = [
    ('gpt_hawk_score', 'GPT-4 Hawkishness'),
    ('bart_hawk_prob', 'BART Hawk Probability'),
    ('finbert_score', 'FinBERT Sentiment'),
    ('hawk_minus_dove', 'Hawk - Dove Count'),
    ('delta_semantic', 'Semantic Change'),
    ('change_overall_similarity', 'Text Similarity Change')
]

for idx, (var, label) in enumerate(nlp_vars):
    ax = axes[idx // 3, idx % 3]

    if var in df.columns:
        plot_df = df[[var, 'dy2_1d_bp']].dropna()

        # Scatter plot
        ax.scatter(plot_df[var], plot_df['dy2_1d_bp'], alpha=0.5, s=30)

        # Trend line
        z = np.polyfit(plot_df[var], plot_df['dy2_1d_bp'], 1)
        p = np.poly1d(z)
        ax.plot(plot_df[var], p(plot_df[var]), "r--", alpha=0.8, linewidth=2)

        # Correlation
        corr = plot_df[var].corr(plot_df['dy2_1d_bp'])
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('2Y Yield Change (bp)', fontsize=10)
        ax.set_title(f'{label} vs Market Reaction', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'{var}\nNot Available', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('plots/figure10_nlp_vs_market_scatter.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/figure10_nlp_vs_market_scatter.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ALL PLOTS CREATED SUCCESSFULLY!")
print("="*70)

print("\n📊 Output files in 'plots/' directory:")
print("  1. figure1_market_reactions_distribution.png - Distribution of yield changes")
print("  2. figure2_market_reactions_timeseries.png - Market reactions over time")
print("  3. figure3_nlp_features_distribution.png - NLP feature distributions")
print("  4. figure4_correlation_heatmap.png - Feature correlation matrix")
print("  5. figure5_change_detection_features.png - Change detection distributions")

if has_models:
    print("  6. figure6_model_comparison_rmse.png - Model RMSE comparison")
    print("  7. figure7_model_comparison_all_metrics.png - All model metrics")

if has_features:
    print("  8. figure8_feature_importance_top20.png - Top 20 important features")
    print("  9. figure9_feature_groups_comparison.png - Feature category comparison")

print(" 10. figure10_nlp_vs_market_scatter.png - NLP features vs market reactions")

print(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("="*70)
