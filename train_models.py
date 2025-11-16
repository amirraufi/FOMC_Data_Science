"""
FOMC Market Reaction Model Training Script

This script:
1. Loads FOMC data and existing NLP features
2. Adds change detection features
3. Computes market reactions
4. Trains multiple models with time-series CV
5. Runs SHAP analysis
6. Generates publication-ready results

Usage:
    python train_models.py

Requirements:
    - communications.csv (FOMC statements and minutes)
    - data_with_gpt_bart_finbert.csv (existing NLP features) [optional]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import shap

# Import our utilities
from fomc_analysis_utils import (
    FOMCDataLoader,
    MarketReactionCalculator,
    ChangeDetector,
    TimeSeriesSplitter
)


def load_and_prepare_data():
    """Load all data and prepare feature matrix"""
    print("="*70)
    print("STEP 1: Loading Data")
    print("="*70)

    # Load FOMC communications
    loader = FOMCDataLoader('communications.csv')
    df = loader.load_communications(start_date='2000-01-01')

    # Separate statements (better for market reactions)
    statements = df[df['Type'] == 'Statement'].copy().reset_index(drop=True)
    print(f"\n✓ Loaded {len(statements)} FOMC statements")

    # Fetch market data
    print("\nFetching market data from FRED...")
    market_df = loader.fetch_market_data()

    # Compute market reactions
    print("\nComputing market reactions...")
    statements = MarketReactionCalculator.compute_reactions(
        statements, market_df, horizons=[1, 2]
    )

    # Add change detection features
    print("\nAdding change detection features...")
    statements = ChangeDetector.add_change_features(statements)

    # Try to load existing NLP features
    try:
        print("\nLoading existing NLP features...")
        existing_features = pd.read_csv('data_with_gpt_bart_finbert.csv')

        # Merge with change features
        print("Merging with change detection features...")
        statements = statements.merge(
            existing_features,
            on='Date',
            how='left',
            suffixes=('', '_existing')
        )
        print(f"✓ Merged features: {statements.shape[1]} total columns")

    except FileNotFoundError:
        print("⚠ data_with_gpt_bart_finbert.csv not found")
        print("  Using only change detection features for now")

    return statements


def prepare_feature_matrix(df, target='dgs2_1d_bp'):
    """
    Prepare features and target for modeling

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    print("\n" + "="*70)
    print("STEP 2: Preparing Feature Matrix")
    print("="*70)

    # Create time-series splits
    splits = TimeSeriesSplitter.create_splits(df, holdout_year=2024, cv_cutoff_year=2017)

    # Define feature columns
    feature_prefixes = ['change_', 'gpt_', 'bart_', 'finbert_']
    feature_cols = []

    for prefix in feature_prefixes:
        feature_cols.extend([col for col in df.columns if col.startswith(prefix)])

    # Add specific features if available
    additional_features = ['hawk_minus_dove', 'delta_semantic', 'is_minute']
    for feat in additional_features:
        if feat in df.columns and feat not in feature_cols:
            feature_cols.append(feat)

    # Remove duplicates and target-related columns
    feature_cols = list(set(feature_cols))
    feature_cols = [col for col in feature_cols if not any([
        'dgs' in col.lower(),
        'dff' in col.lower(),
        'dy' in col.lower(),
        'spread' in col.lower(),
        'bp' in col.lower()
    ])]

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")

    # Prepare data for each split
    def prepare_split(split_df):
        X = split_df[feature_cols].copy()
        y = split_df[target].copy()

        # Handle missing values
        X = X.fillna(0)

        # Filter valid samples
        valid_idx = y.notna()
        return X[valid_idx], y[valid_idx]

    X_train, y_train = prepare_split(splits['train'])
    X_val, y_val = prepare_split(splits['validation'])
    X_test, y_test = prepare_split(splits['holdout'])

    # Also prepare combined train+val for final model
    X_train_val, y_train_val = prepare_split(splits['train_val'])

    print(f"\nData splits:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Holdout: X={X_test.shape}, y={y_test.shape}")
    print(f"\nTarget statistics:")
    print(f"  Train - mean: {y_train.mean():.2f} bp, std: {y_train.std():.2f} bp")
    print(f"  Val - mean: {y_val.mean():.2f} bp, std: {y_val.std():.2f} bp")

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val, feature_cols


def train_models(X_train, X_val, y_train, y_val, feature_names):
    """Train multiple models and compare performance"""
    print("\n" + "="*70)
    print("STEP 3: Training Models")
    print("="*70)

    results = {}
    trained_models = {}

    # Define models
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        ),
        'Gradient Boosting': HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        ),
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)

        # Directional accuracy
        train_dir_acc = np.mean(np.sign(y_pred_train) == np.sign(y_train)) * 100
        val_dir_acc = np.mean(np.sign(y_pred_val) == np.sign(y_val)) * 100

        results[name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_dir_acc': train_dir_acc,
            'val_dir_acc': val_dir_acc,
        }

        trained_models[name] = model

        # Print results
        print(f"\nTrain RMSE: {train_rmse:.3f} bp")
        print(f"Val RMSE:   {val_rmse:.3f} bp")
        print(f"Train MAE:  {train_mae:.3f} bp")
        print(f"Val MAE:    {val_mae:.3f} bp")
        print(f"Train R²:   {train_r2:.3f}")
        print(f"Val R²:     {val_r2:.3f}")
        print(f"Train Dir Acc: {train_dir_acc:.1f}%")
        print(f"Val Dir Acc:   {val_dir_acc:.1f}%")

    # Create comparison table
    results_df = pd.DataFrame(results).T

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(results_df.round(3))

    # Find best model
    best_model_name = results_df['val_rmse'].idxmin()
    print(f"\n🏆 Best model (by validation RMSE): {best_model_name}")
    print(f"   Validation RMSE: {results_df.loc[best_model_name, 'val_rmse']:.3f} bp")
    print(f"   Directional Accuracy: {results_df.loc[best_model_name, 'val_dir_acc']:.1f}%")

    return trained_models, results_df, best_model_name


def run_shap_analysis(model, X_train, X_val, feature_names, model_name):
    """Run SHAP analysis on best model"""
    print("\n" + "="*70)
    print(f"STEP 4: SHAP Analysis for {model_name}")
    print("="*70)

    try:
        # Create SHAP explainer
        if 'Random Forest' in model_name or 'Gradient' in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_val)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\n📊 Top 20 Most Important Features (by mean |SHAP value|):")
        print("="*70)
        print(feature_importance.head(20).to_string(index=False))

        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("\n✓ Saved feature importance to 'feature_importance.csv'")

        # Create SHAP plots
        print("\nGenerating SHAP visualizations...")

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val, feature_names=feature_names,
                         max_display=20, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Saved SHAP summary plot to 'shap_summary_plot.png'")
        plt.close()

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_val, feature_names=feature_names,
                         plot_type='bar', max_display=20, show=False)
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Saved SHAP bar plot to 'shap_bar_plot.png'")
        plt.close()

        return feature_importance, shap_values

    except Exception as e:
        print(f"⚠ SHAP analysis failed: {e}")
        print("  Continuing without SHAP...")
        return None, None


def test_on_holdout(model, X_test, y_test, model_name):
    """Test best model on holdout set"""
    print("\n" + "="*70)
    print(f"STEP 5: Holdout Test (2024-2025) for {model_name}")
    print("="*70)

    if len(X_test) == 0:
        print("⚠ No holdout data available (no FOMC events in 2024-2025 yet)")
        return

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100

    print(f"\nHoldout Set Results:")
    print(f"  RMSE: {rmse:.3f} bp")
    print(f"  MAE:  {mae:.3f} bp")
    print(f"  R²:   {r2:.3f}")
    print(f"  Directional Accuracy: {dir_acc:.1f}%")

    # Create results table
    results_table = pd.DataFrame({
        'Predicted': y_pred,
        'Actual': y_test.values,
        'Error': y_pred - y_test.values
    })

    print("\nPredictions vs Actual:")
    print(results_table)

    return results_table


def generate_publication_figures(results_df):
    """Generate publication-ready figures"""
    print("\n" + "="*70)
    print("STEP 6: Generating Publication Figures")
    print("="*70)

    # Figure 1: Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE comparison
    ax = axes[0, 0]
    results_df[['train_rmse', 'val_rmse']].plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('RMSE (basis points)')
    ax.set_title('Model Comparison: RMSE', fontweight='bold')
    ax.legend(['Training', 'Validation'])
    ax.grid(True, alpha=0.3)

    # R² comparison
    ax = axes[0, 1]
    results_df[['train_r2', 'val_r2']].plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison: R²', fontweight='bold')
    ax.legend(['Training', 'Validation'])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Directional accuracy
    ax = axes[1, 0]
    results_df[['train_dir_acc', 'val_dir_acc']].plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison: Directional Accuracy', fontweight='bold')
    ax.legend(['Training', 'Validation'])
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random Baseline')
    ax.grid(True, alpha=0.3)

    # MAE comparison
    ax = axes[1, 1]
    results_df[['train_mae', 'val_mae']].plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('MAE (basis points)')
    ax.set_title('Model Comparison: MAE', fontweight='bold')
    ax.legend(['Training', 'Validation'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved model comparison figure to 'model_comparison.png'")
    plt.close()

    print("✓ All figures generated successfully!")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("FOMC MARKET REACTION MODEL TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")

    try:
        # Load data
        df = load_and_prepare_data()

        # Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val, feature_names = \
            prepare_feature_matrix(df, target='dgs2_1d_bp')

        # Train models
        models, results_df, best_model_name = train_models(
            X_train, X_val, y_train, y_val, feature_names
        )

        # SHAP analysis on best model
        best_model = models[best_model_name]
        feature_importance, shap_values = run_shap_analysis(
            best_model, X_train, X_val, feature_names, best_model_name
        )

        # Test on holdout
        if len(X_test) > 0:
            holdout_results = test_on_holdout(best_model, X_test, y_test, best_model_name)

        # Generate figures
        generate_publication_figures(results_df)

        # Save results
        results_df.to_csv('model_results.csv')
        print("\n✓ Saved model results to 'model_results.csv'")

        # Final summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nFiles generated:")
        print(f"  - model_results.csv (model comparison)")
        print(f"  - feature_importance.csv (top features)")
        print(f"  - model_comparison.png (publication figure)")
        print(f"  - shap_summary_plot.png (feature importance)")
        print(f"  - shap_bar_plot.png (feature importance bar chart)")

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Validation RMSE: {results_df.loc[best_model_name, 'val_rmse']:.3f} bp")
        print(f"   Directional Accuracy: {results_df.loc[best_model_name, 'val_dir_acc']:.1f}%")

        if feature_importance is not None:
            print(f"\n📊 Top 5 Most Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

        print(f"\n✓ End time: {datetime.now():%Y-%m-%d %H:%M:%S}")

        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Review model_comparison.png for performance")
        print("2. Analyze feature_importance.csv to see what matters")
        print("3. Examine shap_summary_plot.png for interpretability")
        print("4. Use these results in your academic paper!")
        print("5. See RESEARCH_ROADMAP.md for writing guidance")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Required data file not found")
        print(f"   {e}")
        print(f"\n📋 Required files:")
        print(f"   - communications.csv (FOMC statements)")
        print(f"   - data_with_gpt_bart_finbert.csv (optional, for NLP features)")
        print(f"\n💡 Make sure these files are in the current directory and try again.")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
