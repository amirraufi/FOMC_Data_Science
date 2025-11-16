"""
Quick Start Example: FOMC Market Reaction Analysis
Run this to see a simple end-to-end example

Usage:
    python quick_start_example.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fomc_analysis_utils import (
    FOMCDataLoader,
    MarketReactionCalculator,
    ChangeDetector,
    TimeSeriesSplitter,
    create_paper_outline
)

def main():
    print("="*70)
    print("FOMC Market Reaction Analysis - Quick Start Example")
    print("="*70)

    # Step 1: Load FOMC communications
    print("\n[1/6] Loading FOMC communications...")
    loader = FOMCDataLoader('communications.csv')

    try:
        df = loader.load_communications(start_date='2000-01-01')
        statements = df[df['Type'] == 'Statement'].copy()
        print(f"✓ Loaded {len(statements)} FOMC statements")
    except FileNotFoundError:
        print("✗ Error: communications.csv not found")
        print("  Please ensure the data file is in the current directory")
        return

    # Step 2: Fetch market data
    print("\n[2/6] Fetching market data from FRED...")
    try:
        market_df = loader.fetch_market_data()
        print(f"✓ Loaded market data with {len(market_df)} observations")
    except Exception as e:
        print(f"✗ Error fetching market data: {e}")
        return

    # Step 3: Calculate market reactions
    print("\n[3/6] Computing market reactions...")
    statements = MarketReactionCalculator.compute_reactions(
        statements,
        market_df,
        horizons=[1, 2]
    )
    print(f"✓ Computed reactions for {len(statements)} releases")

    # Step 4: Add change detection features (THE NOVEL PART!)
    print("\n[4/6] Adding change detection features...")
    statements = ChangeDetector.add_change_features(statements)

    # Show sample of change features
    change_cols = [col for col in statements.columns if col.startswith('change_')]
    print(f"\nSample change detection features:")
    print(statements[['Date'] + change_cols[:5]].head())

    # Step 5: Create train/validation/holdout splits
    print("\n[5/6] Creating time-series splits...")
    splits = TimeSeriesSplitter.create_splits(
        statements,
        holdout_year=2024,
        cv_cutoff_year=2017
    )

    # Step 6: Basic analysis
    print("\n[6/6] Generating summary statistics...")

    # Market reaction stats
    print("\n" + "="*70)
    print("Market Reaction Statistics (1-day, in basis points)")
    print("="*70)

    reaction_cols = ['dgs2_1d_bp', 'dgs5_1d_bp', 'dgs10_1d_bp', 'dff_1d_bp']
    stats = statements[reaction_cols].describe()
    print(stats)

    # Change feature stats
    print("\n" + "="*70)
    print("Change Detection Feature Statistics")
    print("="*70)

    key_change_features = [
        'change_sentences_added',
        'change_sentences_removed',
        'change_overall_similarity',
        'change_text_length_pct'
    ]

    available_features = [f for f in key_change_features if f in statements.columns]
    if available_features:
        change_stats = statements[available_features].describe()
        print(change_stats)

    # Correlations
    print("\n" + "="*70)
    print("Correlation: Change Features vs Market Reactions")
    print("="*70)

    if available_features:
        # Compute correlation with 2Y yield change
        target = 'dgs2_1d_bp'
        correlations = {}

        for feat in available_features:
            valid_data = statements[[feat, target]].dropna()
            if len(valid_data) > 10:
                corr = valid_data[feat].corr(valid_data[target])
                correlations[feat] = corr

        corr_df = pd.DataFrame.from_dict(
            correlations,
            orient='index',
            columns=['Correlation with 2Y Yield Change']
        ).sort_values('Correlation with 2Y Yield Change', key=abs, ascending=False)

        print(corr_df)

    # Visualization
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # Plot 1: Market reactions over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Panel A: 2Y yield changes
    ax1 = axes[0]
    ax1.bar(statements['Date'], statements['dgs2_1d_bp'],
            color=['red' if x < 0 else 'green' for x in statements['dgs2_1d_bp']],
            alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('2Y Yield Change (bp)')
    ax1.set_title('Market Reactions to FOMC Statements: 2-Year Treasury Yield Changes',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel B: Change features over time
    ax2 = axes[1]
    if 'change_sentences_added' in statements.columns:
        ax2.plot(statements['Date'], statements['change_sentences_added'],
                label='Sentences Added', marker='o', alpha=0.7)
        ax2.plot(statements['Date'], statements['change_sentences_removed'],
                label='Sentences Removed', marker='s', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Sentences')
        ax2.set_title('Change Detection: Sentences Added/Removed Between Statements',
                      fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plt.savefig('fomc_quick_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization to 'fomc_quick_analysis.png'")
    except Exception as e:
        print(f"⚠ Could not save figure: {e}")

    plt.show()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
    ✓ Data loaded successfully
    ✓ Market reactions computed
    ✓ Change detection features added
    ✓ Time-series splits created
    ✓ Basic analysis complete

    Dataset Summary:
    - Total FOMC statements: {len(statements)}
    - Time period: {statements['Date'].min():%Y-%m-%d} to {statements['Date'].max():%Y-%m-%d}
    - Training samples: {len(splits['train'])}
    - Validation samples: {len(splits['validation'])}
    - Holdout samples: {len(splits['holdout'])}

    Change Detection Features: {len([c for c in statements.columns if c.startswith('change_')])}

    Next Steps:
    1. Merge with existing NLP features (GPT-4, FinBERT, BART)
    2. Train models with time-series cross-validation
    3. Run SHAP analysis for interpretability
    4. Generate publication-quality figures
    5. Start writing the paper!

    See ENHANCED_README.md for detailed instructions.
    """)

    # Paper outline
    print("\n" + "="*70)
    print("ACADEMIC PAPER OUTLINE")
    print("="*70)
    create_paper_outline()

    print("\n✓ Quick start example completed successfully!")
    print("  Open FOMC_Enhanced_Research.ipynb to continue the analysis.\n")


if __name__ == "__main__":
    main()
