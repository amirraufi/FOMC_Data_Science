"""
Fetch Intraday Market Data for FOMC Analysis

This script fetches high-frequency market data around FOMC events:
- Treasury ETF prices (minute-level) - FREE via yfinance
- Treasury futures (if available)
- VIX (volatility)
- Dollar index

Usage:
    python fetch_intraday_data.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def fetch_treasury_etf_intraday(ticker, start_date, end_date, interval='1m'):
    """
    Fetch intraday data for Treasury ETFs

    Args:
        ticker: ETF ticker (TLT, IEF, SHY, etc.)
        start_date: Start date
        end_date: End date
        interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d'

    Returns:
        DataFrame with OHLCV data
    """
    try:
        print(f"Fetching {ticker} data ({interval} intervals)...")

        # yfinance limits:
        # 1m data: max 7 days
        # 5m data: max 60 days
        # 15m data: max 60 days

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date, interval=interval)

        if len(df) > 0:
            print(f"  ✓ Got {len(df)} data points for {ticker}")
            return df
        else:
            print(f"  ✗ No data for {ticker}")
            return None

    except Exception as e:
        print(f"  ✗ Error fetching {ticker}: {e}")
        return None


def get_fomc_event_window(event_date, event_time='14:00', hours_before=2, hours_after=4):
    """
    Get time window around FOMC event

    FOMC announcements typically at 2:00 PM ET

    Args:
        event_date: Date of FOMC event (YYYY-MM-DD)
        event_time: Time of announcement (HH:MM)
        hours_before: Hours before to fetch
        hours_after: Hours after to fetch

    Returns:
        start_datetime, end_datetime, event_datetime
    """
    event_datetime = pd.to_datetime(f"{event_date} {event_time}")
    start_datetime = event_datetime - timedelta(hours=hours_before)
    end_datetime = event_datetime + timedelta(hours=hours_after)

    return start_datetime, end_datetime, event_datetime


def fetch_fomc_event_data(event_date, tickers=['TLT', 'IEF', 'SHY'], interval='1m'):
    """
    Fetch intraday data around a specific FOMC event

    Treasury ETFs:
    - TLT: 20+ year Treasury bonds (sensitive to long rates)
    - IEF: 7-10 year Treasury bonds (medium maturity)
    - SHY: 1-3 year Treasury bonds (short maturity)

    Args:
        event_date: FOMC event date
        tickers: List of tickers to fetch
        interval: Data frequency

    Returns:
        Dictionary of DataFrames
    """
    print(f"\n{'='*70}")
    print(f"Fetching data for FOMC event: {event_date}")
    print(f"{'='*70}")

    # Get time window
    start_dt, end_dt, event_dt = get_fomc_event_window(event_date)

    print(f"Event time: {event_dt}")
    print(f"Window: {start_dt} to {end_dt}")
    print(f"Interval: {interval}")

    # For 1-minute data, yfinance has 7-day limit
    # So we fetch just the event day
    if interval == '1m':
        # Get just the event day
        start_date = pd.to_datetime(event_date).date()
        end_date = start_date + timedelta(days=1)
    else:
        start_date = start_dt.date()
        end_date = end_dt.date() + timedelta(days=1)

    data = {}

    for ticker in tickers:
        df = fetch_treasury_etf_intraday(ticker, start_date, end_date, interval)
        if df is not None and len(df) > 0:
            # Filter to event window
            df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]

            if len(df_filtered) > 0:
                data[ticker] = df_filtered
                print(f"  → {ticker}: {len(df_filtered)} data points in event window")
            else:
                # If no data in window, take the full day
                data[ticker] = df
                print(f"  → {ticker}: Using full day data ({len(df)} points)")

    return data, event_dt


def calculate_event_returns(data, event_datetime, windows=[15, 30, 60, 120]):
    """
    Calculate returns in different time windows after FOMC event

    Args:
        data: Dict of DataFrames (from fetch_fomc_event_data)
        event_datetime: Time of FOMC announcement
        windows: List of minutes after event to measure

    Returns:
        DataFrame with returns for each window
    """
    results = []

    for ticker, df in data.items():
        # Get price just before event
        pre_event = df[df.index <= event_datetime]
        if len(pre_event) == 0:
            print(f"  ⚠ {ticker}: No data before event")
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


def fetch_all_fomc_events(fomc_dates, interval='5m'):
    """
    Fetch intraday data for all FOMC events

    Args:
        fomc_dates: List of FOMC dates
        interval: Data frequency ('1m', '5m', '15m', '30m')

    Returns:
        DataFrame with all event returns
    """
    print(f"\n{'='*70}")
    print(f"FETCHING INTRADAY DATA FOR {len(fomc_dates)} FOMC EVENTS")
    print(f"{'='*70}")
    print(f"Interval: {interval}")
    print(f"ETFs: TLT (20Y), IEF (7-10Y), SHY (1-3Y)")

    all_returns = []

    for i, date in enumerate(fomc_dates, 1):
        print(f"\n[{i}/{len(fomc_dates)}] Processing {date}...")

        try:
            # Fetch data
            data, event_dt = fetch_fomc_event_data(
                date,
                tickers=['TLT', 'IEF', 'SHY'],
                interval=interval
            )

            if len(data) == 0:
                print(f"  ⚠ No data available for {date}")
                continue

            # Calculate returns
            returns = calculate_event_returns(data, event_dt)
            returns['date'] = date
            returns['event_datetime'] = event_dt

            all_returns.append(returns)

            # Print summary
            if len(returns) > 0:
                print(f"  ✓ Returns calculated:")
                for _, row in returns.iterrows():
                    ret_15 = row.get('return_15min', np.nan)
                    ret_60 = row.get('return_60min', np.nan)
                    print(f"    {row['ticker']}: 15min={ret_15:.1f}bp, 60min={ret_60:.1f}bp")

        except Exception as e:
            print(f"  ✗ Error processing {date}: {e}")
            continue

    if len(all_returns) > 0:
        results_df = pd.concat(all_returns, ignore_index=True)
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully fetched data for {len(results_df['date'].unique())} events")
        print(f"Total rows: {len(results_df)}")

        return results_df
    else:
        print("\n⚠ No data collected")
        return None


def merge_with_statements(intraday_returns, statements_df):
    """
    Merge intraday returns with FOMC statements

    Args:
        intraday_returns: DataFrame from fetch_all_fomc_events
        statements_df: DataFrame with FOMC statements

    Returns:
        Merged DataFrame
    """
    print(f"\n{'='*70}")
    print("MERGING WITH FOMC STATEMENTS")
    print(f"{'='*70}")

    # Pivot to wide format (one row per event)
    intraday_wide = intraday_returns.pivot_table(
        index='date',
        columns='ticker',
        values=[col for col in intraday_returns.columns if 'return_' in col]
    )

    # Flatten column names
    intraday_wide.columns = [f"{col[1]}_{col[0]}" for col in intraday_wide.columns]
    intraday_wide = intraday_wide.reset_index()

    # Merge with statements
    statements_df['Date'] = pd.to_datetime(statements_df['Date'])
    intraday_wide['date'] = pd.to_datetime(intraday_wide['date'])

    merged = statements_df.merge(
        intraday_wide,
        left_on='Date',
        right_on='date',
        how='left'
    )

    print(f"✓ Merged {len(merged)} statements")
    print(f"  With intraday data: {merged['date'].notna().sum()}")

    # Show sample
    intraday_cols = [col for col in merged.columns if 'return_' in col]
    if len(intraday_cols) > 0:
        print(f"\nSample intraday returns (basis points):")
        print(merged[['Date'] + intraday_cols[:6]].head(10))

    return merged


def main():
    """Main function to fetch intraday data"""

    print("\n" + "="*70)
    print("FOMC INTRADAY MARKET DATA FETCHER")
    print("="*70)

    # Load FOMC statements to get dates
    print("\nLoading FOMC statements...")
    try:
        statements = pd.read_csv('communications.csv')
        statements['Date'] = pd.to_datetime(statements['Date'])

        # Filter to statements only
        statements = statements[statements['Type'] == 'Statement'].copy()

        print(f"✓ Loaded {len(statements)} FOMC statements")
        print(f"  Date range: {statements['Date'].min():%Y-%m-%d} to {statements['Date'].max():%Y-%m-%d}")

    except FileNotFoundError:
        print("✗ communications.csv not found")
        print("  Please make sure the file exists")
        return

    # Get unique FOMC dates
    fomc_dates = sorted(statements['Date'].unique())

    # For testing, use recent events first (better data availability)
    # Filter to 2020 onwards for minute data
    recent_dates = [d for d in fomc_dates if d >= pd.to_datetime('2020-01-01')]

    print(f"\nFOMC events since 2020: {len(recent_dates)}")
    print(f"Sample dates: {recent_dates[:5]}")

    # Ask user which interval to use
    print("\n" + "="*70)
    print("DATA INTERVAL OPTIONS")
    print("="*70)
    print("""
1. 1-minute data (best, but limited to last 7 days per request)
2. 5-minute data (good, up to 60 days per request) ← RECOMMENDED
3. 15-minute data (good, up to 60 days)
4. 30-minute data (less detail)

For academic research, 5-minute data is usually sufficient and more reliable.
    """)

    interval = '5m'  # Default

    # Fetch data for recent events (2020 onwards)
    print(f"\nFetching {interval} data for {len(recent_dates)} recent FOMC events...")
    print("This may take 2-5 minutes...\n")

    intraday_returns = fetch_all_fomc_events(recent_dates, interval=interval)

    if intraday_returns is not None:
        # Save raw intraday returns
        intraday_returns.to_csv('intraday_returns.csv', index=False)
        print(f"\n✓ Saved intraday returns to 'intraday_returns.csv'")

        # Merge with statements
        merged = merge_with_statements(intraday_returns, statements)
        merged.to_csv('statements_with_intraday.csv', index=False)
        print(f"✓ Saved merged data to 'statements_with_intraday.csv'")

        # Summary statistics
        print(f"\n{'='*70}")
        print("INTRADAY RETURN STATISTICS (basis points)")
        print(f"{'='*70}")

        return_cols = [col for col in intraday_returns.columns if 'return_' in col]
        if len(return_cols) > 0:
            stats = intraday_returns[return_cols].describe()
            print(stats.round(2))

        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
        print(f"""
Files created:
  - intraday_returns.csv ({len(intraday_returns)} rows)
  - statements_with_intraday.csv ({len(merged)} rows)

Next steps:
  1. Review the data in Excel or pandas
  2. Update train_models.py to use intraday returns
  3. Compare: Do 15-min returns predict better than daily?
  4. Add to your paper: "High-frequency analysis (5-minute intervals)"

Example usage in paper:
  "We measure market reactions using 5-minute Treasury ETF price data,
   calculating returns in 15-minute, 30-minute, and 60-minute windows
   following FOMC statement releases."
        """)

    else:
        print("\n⚠ No intraday data collected")
        print("  Possible reasons:")
        print("  - Too old events (yfinance has limited history)")
        print("  - Network issues")
        print("  - Try with more recent events only")


if __name__ == "__main__":
    main()
