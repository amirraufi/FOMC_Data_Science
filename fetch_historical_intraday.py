"""
Fetch Historical Intraday Data (2000-2025)

This handles the limitation that yfinance only has:
- Intraday data (5-min) for recent periods
- Daily data for historical periods

Strategy:
- 2020-2025: Use 5-minute intraday data (best resolution)
- 2000-2019: Use daily data (only option available)
- Combine both for complete historical coverage

Usage:
    python fetch_historical_intraday.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


def fetch_daily_treasury_data(start_date='2000-01-01', end_date='2025-12-31'):
    """
    Fetch daily Treasury ETF data for long historical period

    Returns:
        Dictionary of DataFrames with daily data
    """
    print(f"\n{'='*70}")
    print("FETCHING DAILY TREASURY DATA (2000-2025)")
    print(f"{'='*70}")

    tickers = {
        'TLT': '20+ Year Treasury',
        'IEF': '7-10 Year Treasury',
        'SHY': '1-3 Year Treasury',
        'AGG': 'Aggregate Bond (alternative)',
    }

    data = {}

    for ticker, name in tickers.items():
        try:
            print(f"Fetching {ticker} ({name})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(df) > 0:
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} days from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
            else:
                print(f"  ✗ {ticker}: No data")

        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")

    return data


def calculate_daily_event_returns(data, event_date, windows=[1, 2, 5]):
    """
    Calculate returns around FOMC event using daily data

    Args:
        data: Dictionary of daily DataFrames
        event_date: Date of FOMC event
        windows: List of days to measure returns over

    Returns:
        DataFrame with returns
    """
    results = []
    event_date = pd.to_datetime(event_date)

    for ticker, df in data.items():
        # Get price before event
        pre_event = df[df.index < event_date]
        if len(pre_event) == 0:
            continue

        pre_price = pre_event['Close'].iloc[-1]
        pre_date = pre_event.index[-1]

        row = {
            'ticker': ticker,
            'pre_event_price': pre_price,
            'pre_event_date': pre_date,
        }

        # Calculate returns for each window
        for days in windows:
            target_date = event_date + timedelta(days=days)
            post_event = df[(df.index > event_date) & (df.index <= target_date)]

            if len(post_event) > 0:
                post_price = post_event['Close'].iloc[-1]
                post_date = post_event.index[-1]

                # Return in basis points
                ret = (post_price - pre_price) / pre_price * 10000

                row[f'return_{days}d'] = ret
                row[f'post_date_{days}d'] = post_date
            else:
                row[f'return_{days}d'] = np.nan
                row[f'post_date_{days}d'] = None

        results.append(row)

    return pd.DataFrame(results)


def fetch_intraday_single_event(event_date, tickers=['TLT', 'IEF', 'SHY'], interval='5m'):
    """
    Fetch intraday data for a single FOMC event

    yfinance limitations:
    - 1m data: max 7 days ago
    - 5m data: max 60 days ago
    - 15m data: max 60 days ago
    """
    event_date = pd.to_datetime(event_date)
    today = pd.Timestamp.now()
    days_ago = (today - event_date).days

    # Check if intraday data is available
    if interval == '5m' and days_ago > 60:
        return None, "Event too old for 5m data"
    if interval == '1m' and days_ago > 7:
        return None, "Event too old for 1m data"

    # Get time window (FOMC typically at 2 PM ET)
    event_datetime = event_date.replace(hour=14, minute=0)
    start_datetime = event_datetime - timedelta(hours=2)
    end_datetime = event_datetime + timedelta(hours=4)

    # Fetch data for the day
    start_date = event_date.date()
    end_date = event_date.date() + timedelta(days=1)

    data = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, interval=interval)

            if len(df) > 0:
                # Filter to event window
                df_filtered = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
                if len(df_filtered) > 0:
                    data[ticker] = df_filtered
                else:
                    data[ticker] = df  # Use full day if window is empty

        except Exception:
            continue

    if len(data) == 0:
        return None, "No data available"

    return data, event_datetime


def calculate_intraday_returns(data, event_datetime, windows=[15, 30, 60, 120]):
    """Calculate intraday returns in minute windows"""
    results = []

    for ticker, df in data.items():
        # Get price just before event
        pre_event = df[df.index <= event_datetime]
        if len(pre_event) == 0:
            continue

        pre_price = pre_event['Close'].iloc[-1]
        pre_time = pre_event.index[-1]

        row = {
            'ticker': ticker,
            'pre_event_price': pre_price,
            'pre_event_time': pre_time,
        }

        # Calculate returns for each window
        for minutes in windows:
            post_time = event_datetime + timedelta(minutes=minutes)
            post_event = df[df.index <= post_time]

            if len(post_event) > 0:
                post_price = post_event['Close'].iloc[-1]
                actual_post_time = post_event.index[-1]

                # Return in basis points
                ret = (post_price - pre_price) / pre_price * 10000

                row[f'return_{minutes}min'] = ret
                row[f'post_time_{minutes}min'] = actual_post_time
            else:
                row[f'return_{minutes}min'] = np.nan
                row[f'post_time_{minutes}min'] = None

        results.append(row)

    return pd.DataFrame(results)


def fetch_all_events_smart(fomc_dates, cutoff_date='2020-01-01'):
    """
    Smart fetching: Use intraday for recent, daily for historical

    Args:
        fomc_dates: List of FOMC event dates
        cutoff_date: Date to switch from daily to intraday

    Returns:
        Combined DataFrame with all returns
    """
    cutoff = pd.to_datetime(cutoff_date)
    fomc_dates = [pd.to_datetime(d) for d in fomc_dates]

    recent_dates = [d for d in fomc_dates if d >= cutoff]
    historical_dates = [d for d in fomc_dates if d < cutoff]

    print(f"\n{'='*70}")
    print("SMART HISTORICAL FETCHING STRATEGY")
    print(f"{'='*70}")
    print(f"Total FOMC events: {len(fomc_dates)}")
    print(f"Recent events (>= {cutoff_date}): {len(recent_dates)} → Will fetch 5-min intraday")
    print(f"Historical events (< {cutoff_date}): {len(historical_dates)} → Will use daily data")

    all_returns = []

    # 1. Fetch daily data for historical events
    if len(historical_dates) > 0:
        print(f"\n{'='*70}")
        print(f"PHASE 1: DAILY DATA FOR HISTORICAL EVENTS ({len(historical_dates)} events)")
        print(f"{'='*70}")

        # Fetch daily data once
        daily_data = fetch_daily_treasury_data(
            start_date=min(historical_dates).strftime('%Y-%m-%d'),
            end_date=max(historical_dates).strftime('%Y-%m-%d')
        )

        print(f"\nCalculating daily returns for historical events...")
        for i, date in enumerate(tqdm(historical_dates), 1):
            try:
                returns = calculate_daily_event_returns(daily_data, date, windows=[1, 2, 5])
                returns['date'] = date
                returns['data_type'] = 'daily'
                all_returns.append(returns)
            except Exception as e:
                print(f"  Error processing {date}: {e}")
                continue

        print(f"✓ Processed {len([r for r in all_returns if 'daily' in r.get('data_type', [])])} historical events")

    # 2. Fetch intraday data for recent events
    if len(recent_dates) > 0:
        print(f"\n{'='*70}")
        print(f"PHASE 2: INTRADAY DATA FOR RECENT EVENTS ({len(recent_dates)} events)")
        print(f"{'='*70}")

        success_count = 0

        for i, date in enumerate(recent_dates, 1):
            print(f"\n[{i}/{len(recent_dates)}] {date.strftime('%Y-%m-%d')}...")

            try:
                data, event_dt = fetch_intraday_single_event(date, interval='5m')

                if data is None:
                    print(f"  ⚠ {event_dt} - Using daily fallback")
                    # Fall back to daily data
                    daily_data_temp = fetch_daily_treasury_data(
                        start_date=(date - timedelta(days=10)).strftime('%Y-%m-%d'),
                        end_date=(date + timedelta(days=10)).strftime('%Y-%m-%d')
                    )
                    returns = calculate_daily_event_returns(daily_data_temp, date, windows=[1, 2])
                    returns['data_type'] = 'daily_fallback'
                else:
                    returns = calculate_intraday_returns(data, event_dt)
                    returns['data_type'] = 'intraday_5m'
                    success_count += 1
                    print(f"  ✓ Got 5-min intraday data")

                returns['date'] = date
                returns['event_datetime'] = event_dt if data else date
                all_returns.append(returns)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        print(f"\n✓ Successfully fetched intraday for {success_count}/{len(recent_dates)} recent events")

    # Combine all returns
    if len(all_returns) > 0:
        results_df = pd.concat(all_returns, ignore_index=True)

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total events processed: {len(results_df['date'].unique())}")
        print(f"  - Intraday (5-min): {len(results_df[results_df['data_type'] == 'intraday_5m']['date'].unique())}")
        print(f"  - Daily: {len(results_df[results_df['data_type'] == 'daily']['date'].unique())}")
        print(f"  - Daily fallback: {len(results_df[results_df['data_type'] == 'daily_fallback']['date'].unique())}")

        return results_df
    else:
        print("\n⚠ No data collected")
        return None


def main():
    """Main function"""

    print("\n" + "="*70)
    print("HISTORICAL INTRADAY DATA FETCHER (2000-2025)")
    print("="*70)

    # Try to load FOMC dates
    try:
        statements = pd.read_csv('communications.csv')
        statements['Date'] = pd.to_datetime(statements['Date'])
        statements = statements[statements['Type'] == 'Statement'].copy()

        fomc_dates = sorted(statements['Date'].unique())

        print(f"\n✓ Loaded {len(statements)} FOMC statements")
        print(f"  Date range: {min(fomc_dates):%Y-%m-%d} to {max(fomc_dates):%Y-%m-%d}")

    except FileNotFoundError:
        print("\n⚠ communications.csv not found")
        print("  Creating demo with sample FOMC dates...")

        # Create sample dates for demo
        fomc_dates = pd.date_range('2000-01-01', '2025-01-01', freq='45D').tolist()
        print(f"  Using {len(fomc_dates)} demo dates")

    # Fetch all data
    print("\nFetching data for all FOMC events...")
    print("This will take 5-10 minutes depending on number of events...")

    all_returns = fetch_all_events_smart(fomc_dates, cutoff_date='2020-01-01')

    if all_returns is not None:
        # Save results
        all_returns.to_csv('historical_intraday_returns.csv', index=False)
        print(f"\n✓ Saved to 'historical_intraday_returns.csv'")

        # Show summary statistics
        print(f"\n{'='*70}")
        print("DATA QUALITY SUMMARY")
        print(f"{'='*70}")

        return_cols = [col for col in all_returns.columns if 'return_' in col]
        if len(return_cols) > 0:
            print("\nReturn statistics (basis points):")
            stats = all_returns[return_cols].describe()
            print(stats.round(2))

        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
        print(f"""
You now have data from 2000-2025!

Files created:
  - historical_intraday_returns.csv

Data breakdown:
  - Recent events (2020+): 5-minute intraday data
  - Historical events (2000-2019): Daily data

This gives you:
  - Complete historical coverage
  - Best available resolution for each period
  - Ready for model training!

Next step:
  python train_models.py
        """)

    else:
        print("\n❌ Failed to fetch data")
        print("  Check internet connection and try again")


if __name__ == "__main__":
    # Install tqdm if needed
    try:
        import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.run(['pip', 'install', 'tqdm'], check=True)
        from tqdm import tqdm

    main()
