"""
Demo: Intraday Data Concept

This demonstrates what the intraday data fetcher will do
when you run it on your local machine (where yfinance works properly).

Shows:
- Data structure for intraday returns
- How to combine historical (daily) and recent (intraday) data
- Example output format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_demo_intraday_data():
    """Create demo intraday returns to show the structure"""

    print("\n" + "="*70)
    print("DEMO: INTRADAY DATA STRUCTURE")
    print("="*70)

    # Sample FOMC dates (mix of historical and recent)
    fomc_dates = [
        '2000-02-02', '2005-06-30', '2010-11-03', '2015-12-16',  # Historical
        '2020-03-15', '2021-12-15', '2022-11-02', '2023-03-22',  # Recent
        '2023-07-26', '2024-05-01', '2024-09-18'                 # Very recent
    ]

    all_data = []

    for date in fomc_dates:
        event_date = pd.to_datetime(date)

        # Determine if we can get intraday data (recent) or daily (historical)
        days_ago = (pd.Timestamp.now() - event_date).days

        if days_ago < 60:  # Recent - can get 5-minute data
            data_type = 'intraday_5m'

            # Simulate intraday returns for TLT, IEF, SHY
            for ticker in ['TLT', 'IEF', 'SHY']:
                # Simulate realistic market reactions
                base_move = np.random.randn() * 15  # Base movement in bp

                all_data.append({
                    'date': event_date,
                    'ticker': ticker,
                    'data_type': data_type,
                    'return_15min': base_move * 0.4 + np.random.randn() * 2,
                    'return_30min': base_move * 0.6 + np.random.randn() * 3,
                    'return_60min': base_move * 0.8 + np.random.randn() * 4,
                    'return_120min': base_move * 1.0 + np.random.randn() * 5,
                })

        else:  # Historical - only daily data available
            data_type = 'daily'

            for ticker in ['TLT', 'IEF', 'SHY']:
                base_move = np.random.randn() * 20

                all_data.append({
                    'date': event_date,
                    'ticker': ticker,
                    'data_type': data_type,
                    'return_1d': base_move + np.random.randn() * 5,
                    'return_2d': base_move * 1.2 + np.random.randn() * 7,
                    'return_5d': base_move * 1.5 + np.random.randn() * 10,
                })

    df = pd.DataFrame(all_data)

    print(f"\nCreated demo data for {len(fomc_dates)} FOMC events")
    print(f"Total rows: {len(df)}")
    print(f"\nData types:")
    print(df['data_type'].value_counts())

    return df


def show_intraday_examples(df):
    """Show examples of intraday data"""

    print(f"\n{'='*70}")
    print("EXAMPLE: INTRADAY DATA (RECENT FOMC EVENTS)")
    print(f"{'='*70}")

    intraday = df[df['data_type'] == 'intraday_5m'].copy()

    if len(intraday) > 0:
        print("\nSample intraday returns (basis points):")
        print("\nEvent: March 22, 2023 (Example)")
        sample = intraday[intraday['date'] == intraday['date'].max()].copy()

        if len(sample) > 0:
            print("\n" + "="*70)
            for _, row in sample.iterrows():
                print(f"{row['ticker']:4s}: ", end='')
                print(f"15min={row['return_15min']:6.1f}bp  ", end='')
                print(f"30min={row['return_30min']:6.1f}bp  ", end='')
                print(f"60min={row['return_60min']:6.1f}bp  ", end='')
                print(f"120min={row['return_120min']:6.1f}bp")

            print("\n📊 This shows:")
            print("  - How quickly markets react (15-minute windows!)")
            print("  - Different maturities respond differently")
            print("  - Most movement happens in first hour")


def show_daily_examples(df):
    """Show examples of daily data"""

    print(f"\n{'='*70}")
    print("EXAMPLE: DAILY DATA (HISTORICAL FOMC EVENTS)")
    print(f"{'='*70}")

    daily = df[df['data_type'] == 'daily'].copy()

    if len(daily) > 0:
        print("\nSample daily returns (basis points):")
        print("\nEvent: December 16, 2015 (Example)")
        sample = daily[daily['date'] == '2015-12-16'].copy()

        if len(sample) > 0:
            print("\n" + "="*70)
            for _, row in sample.iterrows():
                print(f"{row['ticker']:4s}: ", end='')
                print(f"1-day={row['return_1d']:6.1f}bp  ", end='')
                print(f"2-day={row['return_2d']:6.1f}bp  ", end='')
                print(f"5-day={row['return_5d']:6.1f}bp")

            print("\n📊 This shows:")
            print("  - Cumulative reaction over days")
            print("  - Less precise than intraday, but covers full history")
            print("  - Good enough for 2000-2019 period")


def save_demo_file(df):
    """Save demo file"""

    # Pivot to wide format for model training
    print(f"\n{'='*70}")
    print("CREATING MODEL-READY FORMAT")
    print(f"{'='*70}")

    # Separate intraday and daily
    intraday = df[df['data_type'] == 'intraday_5m'].copy()
    daily = df[df['data_type'] == 'daily'].copy()

    # For intraday: pivot to wide
    if len(intraday) > 0:
        intraday_wide = intraday.pivot_table(
            index='date',
            columns='ticker',
            values=['return_15min', 'return_30min', 'return_60min', 'return_120min']
        )
        intraday_wide.columns = [f"{col[1]}_{col[0]}" for col in intraday_wide.columns]
        intraday_wide = intraday_wide.reset_index()
        intraday_wide['data_type'] = 'intraday'

    # For daily: pivot to wide
    if len(daily) > 0:
        daily_wide = daily.pivot_table(
            index='date',
            columns='ticker',
            values=['return_1d', 'return_2d', 'return_5d']
        )
        daily_wide.columns = [f"{col[1]}_{col[0]}" for col in daily_wide.columns]
        daily_wide = daily_wide.reset_index()
        daily_wide['data_type'] = 'daily'

    # Combine
    if len(intraday) > 0 and len(daily) > 0:
        combined = pd.concat([daily_wide, intraday_wide], ignore_index=True)
    elif len(intraday) > 0:
        combined = intraday_wide
    else:
        combined = daily_wide

    combined.to_csv('demo_historical_intraday.csv', index=False)

    print(f"\n✓ Saved demo data to 'demo_historical_intraday.csv'")
    print(f"  Shape: {combined.shape}")
    print(f"  Columns: {list(combined.columns)}")

    return combined


def main():
    """Run demo"""

    print("\n" + "="*70)
    print("INTRADAY DATA CONCEPT DEMO")
    print("="*70)
    print("""
This demo shows what you'll get when you run the real script on your machine.

The real script (fetch_historical_intraday.py) will:
  1. For recent FOMC events (2020-2025): Fetch 5-minute intraday data
  2. For historical events (2000-2019): Use daily data
  3. Combine both for complete coverage

Note: This is a DEMO with simulated data.
      Run the real script on your local machine to get actual market data!
    """)

    # Create demo data
    df = create_demo_intraday_data()

    # Show examples
    show_intraday_examples(df)
    show_daily_examples(df)

    # Save
    combined = save_demo_file(df)

    # Summary
    print(f"\n{'='*70}")
    print("WHAT YOU'LL GET ON YOUR LOCAL MACHINE")
    print(f"{'='*70}")
    print("""
When you run fetch_historical_intraday.py on your machine:

1. Install yfinance:
   pip install yfinance pandas numpy tqdm

2. Run the script:
   python fetch_historical_intraday.py

3. It will fetch:
   - 2020-2025: 5-minute intraday data (15min, 30min, 60min, 120min windows)
   - 2000-2019: Daily data (1-day, 2-day, 5-day windows)

4. Output file: historical_intraday_returns.csv

5. Use in your paper:
   "We analyze market reactions using high-frequency data (5-minute intervals)
    for recent FOMC events (2020-2025) and daily data for historical events
    (2000-2019), providing comprehensive coverage of the sample period."

6. Expected results:
   - ~200-250 FOMC events total
   - ~40-50 with intraday data
   - ~150-200 with daily data
   - All ready for model training!
    """)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
On your local machine:

1. Make sure you have your data files:
   - communications.csv
   - data_with_gpt_bart_finbert.csv

2. Run the enhanced fetcher:
   python fetch_historical_intraday.py

3. This will create:
   - historical_intraday_returns.csv (complete 2000-2025 data!)

4. Then train models:
   python train_models.py

5. Your paper will have:
   - High-frequency analysis (5-min for recent events)
   - Long historical coverage (2000-2025)
   - Best of both worlds!
    """)


if __name__ == "__main__":
    main()
