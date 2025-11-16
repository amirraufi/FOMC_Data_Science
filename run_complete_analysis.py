"""
Complete FOMC Analysis Pipeline
Runs everything: Fetch intraday data → Train models → Generate results

Usage:
    python run_complete_analysis.py
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False


def check_files_exist():
    """Check if required files exist"""
    required_files = {
        'communications.csv': 'FOMC communications data',
        'fomc_analysis_utils.py': 'Utility functions'
    }

    optional_files = {
        'data_with_gpt_bart_finbert.csv': 'NLP features (optional)',
        'intraday_returns.csv': 'Intraday data (will be created)'
    }

    print(f"\n{'='*70}")
    print("CHECKING REQUIRED FILES")
    print(f"{'='*70}")

    all_good = True

    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} - {description} ({size:,} bytes)")
        else:
            print(f"✗ {filename} - MISSING! - {description}")
            all_good = False

    print(f"\nOptional files:")
    for filename, description in optional_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} - {description} ({size:,} bytes)")
        else:
            print(f"  {filename} - Not found (will create if needed)")

    return all_good


def main():
    """Run complete analysis pipeline"""

    print("\n" + "="*70)
    print("FOMC COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    print("""
This script runs the complete analysis:
1. Fetch intraday market data (5-minute intervals)
2. Train models with all features
3. Run SHAP analysis
4. Generate publication figures

Estimated time: 10-15 minutes
    """)

    # Check files
    if not check_files_exist():
        print("\n❌ Missing required files!")
        print("\nPlease ensure you have:")
        print("  - communications.csv (your FOMC data)")
        print("\nSee DATA_PREPARATION_GUIDE.md for help")
        return

    # Ask to continue
    print(f"\n{'='*70}")
    try:
        response = input("Continue with analysis? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    except EOFError:
        print("\n(Non-interactive mode - proceeding automatically)")

    # Step 1: Fetch intraday data
    print("\n" + "="*70)
    print("PHASE 1: FETCH INTRADAY DATA")
    print("="*70)

    if os.path.exists('intraday_returns.csv'):
        print("✓ intraday_returns.csv already exists")
        print("  Skipping data fetch (delete file to re-fetch)")
    else:
        success = run_command(
            f"{sys.executable} fetch_intraday_data.py",
            "Fetching 5-minute market data"
        )

        if not success:
            print("\n⚠ Intraday data fetch failed")
            print("  Continuing with daily data only...")

    # Step 2: Train models
    print("\n" + "="*70)
    print("PHASE 2: TRAIN MODELS")
    print("="*70)

    success = run_command(
        f"{sys.executable} train_models.py",
        "Training models with cross-validation"
    )

    if not success:
        print("\n❌ Model training failed!")
        print("  Check error messages above")
        return

    # Step 3: Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE! 🎉")
    print("="*70)

    print("\nFiles generated:")
    output_files = [
        ('model_results.csv', 'Model performance comparison'),
        ('feature_importance.csv', 'Top features by SHAP'),
        ('model_comparison.png', 'Performance visualization'),
        ('shap_summary_plot.png', 'Feature importance plot'),
        ('intraday_returns.csv', 'High-frequency market data'),
    ]

    for filename, description in output_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename} - {description} ({size:,} bytes)")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("""
1. Review Results:
   - Open model_comparison.png
   - Check feature_importance.csv
   - Review shap_summary_plot.png

2. Check Performance:
   - Look for >60% directional accuracy
   - Verify change features are in top 10

3. Start Writing Paper:
   - Use RESEARCH_ROADMAP.md for structure
   - Include generated figures
   - Report results from model_results.csv

4. If results are good:
   - Draft methodology section
   - Write results section
   - Prepare for submission!
    """)

    print("\n✓ Complete analysis pipeline finished successfully!")


if __name__ == "__main__":
    main()
