"""
Data Validation Script

This script validates all data sources and checks data quality.
Run this after fetching data to ensure everything is correct.

Usage:
    python validate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def check_file_exists(filename):
    """Check if file exists"""
    import os
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  ✓ {filename} exists ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {filename} NOT FOUND")
        return False


def validate_communications_csv():
    """Validate FOMC communications data"""
    print("\n" + "="*70)
    print("1. VALIDATING FOMC COMMUNICATIONS DATA")
    print("="*70)

    if not check_file_exists('communications.csv'):
        print("  ⚠ Cannot validate - file missing")
        return False

    try:
        df = pd.read_csv('communications.csv')
        print(f"\n  Loaded: {len(df)} rows")

        # Check required columns
        required_cols = ['Date', 'Release Date', 'Type', 'Text']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            print(f"  Found columns: {df.columns.tolist()}")
            return False
        else:
            print(f"  ✓ All required columns present")

        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df['Release Date'] = pd.to_datetime(df['Release Date'])

        # Check date range
        print(f"\n  Date range: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

        # Check document types
        print(f"\n  Document types:")
        print(df['Type'].value_counts().to_string())

        # Check for missing text
        missing_text = df['Text'].isna().sum()
        print(f"\n  Missing text: {missing_text} documents")

        # Check text lengths
        df['text_length'] = df['Text'].str.len()
        print(f"\n  Text length stats:")
        print(f"    Mean: {df['text_length'].mean():.0f} characters")
        print(f"    Min: {df['text_length'].min():.0f} characters")
        print(f"    Max: {df['text_length'].max():.0f} characters")

        # Warnings
        if df['text_length'].min() < 100:
            print(f"  ⚠ Warning: Some texts are very short (<100 chars)")

        print("\n  ✓ Communications data looks good!")
        return True

    except Exception as e:
        print(f"  ✗ Error validating: {e}")
        return False


def validate_nlp_features():
    """Validate NLP features data"""
    print("\n" + "="*70)
    print("2. VALIDATING NLP FEATURES DATA")
    print("="*70)

    if not check_file_exists('data_with_gpt_bart_finbert.csv'):
        print("  ⚠ File not found - this is optional")
        return True  # Optional file

    try:
        df = pd.read_csv('data_with_gpt_bart_finbert.csv')
        print(f"\n  Loaded: {len(df)} rows")
        print(f"  Columns: {len(df.columns)}")

        # Check for common NLP feature columns
        nlp_prefixes = ['gpt', 'bart', 'finbert', 'hawk', 'delta']
        nlp_cols = [col for col in df.columns for prefix in nlp_prefixes if prefix in col.lower()]

        print(f"\n  Found {len(nlp_cols)} NLP feature columns:")
        for col in nlp_cols[:10]:
            print(f"    - {col}")
        if len(nlp_cols) > 10:
            print(f"    ... and {len(nlp_cols)-10} more")

        # Check for missing values
        missing = df[nlp_cols].isna().sum()
        if missing.sum() > 0:
            print(f"\n  Missing values:")
            print(missing[missing > 0].to_string())
        else:
            print(f"\n  ✓ No missing values in NLP features")

        # Check value ranges
        for col in nlp_cols[:5]:
            if df[col].dtype in [np.float64, np.int64]:
                print(f"\n  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

        print("\n  ✓ NLP features data looks good!")
        return True

    except Exception as e:
        print(f"  ✗ Error validating: {e}")
        return False


def validate_market_data():
    """Validate fetched market data"""
    print("\n" + "="*70)
    print("3. VALIDATING MARKET DATA (FRED)")
    print("="*70)

    try:
        # Try to fetch a small sample
        print("\n  Testing FRED data access...")
        import pandas_datareader.data as web

        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        test_series = {'DGS2': '2-Year Treasury'}

        for code, name in test_series.items():
            try:
                data = web.DataReader(code, 'fred', start_date, end_date)
                print(f"  ✓ {name} ({code}): {len(data)} observations")
                print(f"    Latest value: {data[code].iloc[-1]:.2f}%")
                print(f"    Date: {data.index[-1]:%Y-%m-%d}")
            except Exception as e:
                print(f"  ✗ {name} ({code}): Failed - {e}")
                return False

        print("\n  ✓ FRED data access working!")
        return True

    except ImportError:
        print("  ✗ pandas_datareader not installed")
        print("    Install: pip install pandas-datareader")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_intraday_data():
    """Validate intraday data if available"""
    print("\n" + "="*70)
    print("4. VALIDATING INTRADAY DATA")
    print("="*70)

    filename = 'historical_intraday_returns.csv'

    if not check_file_exists(filename):
        print("  ⚠ File not found - run fetch_historical_intraday.py first")
        return True  # Not an error, just not done yet

    try:
        df = pd.read_csv(filename)
        print(f"\n  Loaded: {len(df)} rows")

        # Check data types
        print(f"\n  Data types:")
        if 'data_type' in df.columns:
            print(df['data_type'].value_counts().to_string())

        # Check date range
        df['date'] = pd.to_datetime(df['date'])
        print(f"\n  Date range: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

        # Check tickers
        if 'ticker' in df.columns:
            print(f"\n  Tickers: {df['ticker'].unique().tolist()}")

        # Check return columns
        return_cols = [col for col in df.columns if 'return_' in col]
        print(f"\n  Return columns: {return_cols}")

        # Statistics
        print(f"\n  Return statistics (basis points):")
        if len(return_cols) > 0:
            stats = df[return_cols].describe()
            print(stats.round(2))

        # Check for outliers
        for col in return_cols:
            outliers = df[np.abs(df[col]) > 100].shape[0]
            if outliers > 0:
                print(f"\n  ⚠ {col}: {outliers} returns > 100bp (possible outliers)")

        # Check for missing values
        missing = df[return_cols].isna().sum()
        if missing.sum() > 0:
            print(f"\n  Missing values:")
            print(missing[missing > 0].to_string())

        print("\n  ✓ Intraday data looks good!")
        return True

    except Exception as e:
        print(f"  ✗ Error validating: {e}")
        return False


def validate_change_features():
    """Validate change detection features"""
    print("\n" + "="*70)
    print("5. VALIDATING CHANGE DETECTION FEATURES")
    print("="*70)

    # These are generated by the analysis scripts
    print("\n  Change features are generated during analysis")
    print("  Run train_models.py or use fomc_analysis_utils.py")

    # Try to generate a small sample
    try:
        from fomc_analysis_utils import ChangeDetector

        print("\n  Testing change detection...")

        text1 = "The Federal Reserve has decided to maintain the target range. Inflation remains elevated."
        text2 = "The Federal Reserve has decided to raise the target range. Inflation has moderated."

        changes = ChangeDetector.detect_changes(text2, text1)

        print(f"  ✓ Change detection working!")
        print(f"    Features generated: {len(changes)}")
        print(f"    Sample features:")
        for key, value in list(changes.items())[:5]:
            print(f"      {key}: {value}")

        return True

    except ImportError:
        print("  ✗ fomc_analysis_utils not available")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_data_integration():
    """Test that all data can be integrated"""
    print("\n" + "="*70)
    print("6. TESTING DATA INTEGRATION")
    print("="*70)

    try:
        # Check if we can load and merge data
        files_needed = {
            'communications.csv': False,
            'data_with_gpt_bart_finbert.csv': True,  # Optional
        }

        all_exist = True
        for filename, optional in files_needed.items():
            exists = check_file_exists(filename)
            if not exists and not optional:
                all_exist = False

        if not all_exist:
            print("\n  ⚠ Cannot test integration - missing required files")
            return False

        # Try to load communications
        df = pd.read_csv('communications.csv')
        df['Date'] = pd.to_datetime(df['Date'])

        print(f"\n  Base data: {len(df)} FOMC events")

        # Try to add change features
        from fomc_analysis_utils import ChangeDetector

        df_with_changes = ChangeDetector.add_change_features(df)
        change_cols = [col for col in df_with_changes.columns if col.startswith('change_')]

        print(f"  Added {len(change_cols)} change features")

        # Try to merge with NLP features if available
        try:
            nlp = pd.read_csv('data_with_gpt_bart_finbert.csv')
            nlp['Date'] = pd.to_datetime(nlp['Date'])

            merged = df_with_changes.merge(nlp, on='Date', how='left', suffixes=('', '_nlp'))
            print(f"  Merged with NLP features: {merged.shape}")

        except:
            print(f"  NLP features not available (optional)")

        print("\n  ✓ Data integration successful!")
        return True

    except Exception as e:
        print(f"  ✗ Error in integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_quality_report():
    """Generate overall data quality report"""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)

    results = {
        'Communications Data': validate_communications_csv(),
        'NLP Features': validate_nlp_features(),
        'Market Data (FRED)': validate_market_data(),
        'Intraday Data': validate_intraday_data(),
        'Change Features': validate_change_features(),
        'Data Integration': test_data_integration(),
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\n{passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✓✓✓ ALL DATA VALIDATED SUCCESSFULLY! ✓✓✓")
        print("\nYou're ready to:")
        print("  1. Run: python fetch_historical_intraday.py")
        print("  2. Run: python train_models.py")
        print("  3. Start writing your paper!")
    else:
        print("\n⚠ Some validation issues found")
        print("\nRecommended fixes:")
        if not results['Communications Data']:
            print("  - Add communications.csv to the directory")
        if not results['Market Data (FRED)']:
            print("  - Install pandas-datareader: pip install pandas-datareader")
        if not results['Intraday Data']:
            print("  - Run: python fetch_historical_intraday.py")
        if not results['Change Features']:
            print("  - Check fomc_analysis_utils.py is present")

    return all(results.values())


def main():
    """Run all validations"""
    print("\n" + "="*70)
    print("DATA VALIDATION & QUALITY CHECK")
    print("="*70)
    print(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    success = generate_quality_report()

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return success


if __name__ == "__main__":
    main()
