import pandas as pd
import numpy as np

def generate_backup_data():
    print("Generating local backup market data (Jan 2015 - June 2026)...")
    date_range = pd.date_range(start="2015-01-01", end="2026-06-30", freq="B") # Business days
    
    np.random.seed(42)
    n_days = len(date_range)
    
    # Generate realistic geometric Brownian motion steps for returns
    tsla_returns = np.random.normal(0.0012, 0.035, n_days)  # High return, extreme risk
    bnd_returns = np.random.normal(0.0001, 0.004, n_days)   # Flat return, ultra-low risk
    spy_returns = np.random.normal(0.0005, 0.012, n_days)   # Moderate growth, market risk
    
    # Build price paths starting from real 2015 baseline handles
    tsla_prices = 15.0 * np.exp(np.cumsum(tsla_returns))
    bnd_prices = 80.0 * np.exp(np.cumsum(bnd_returns))
    spy_prices = 200.0 * np.exp(np.cumsum(spy_returns))
    
    # Structure MultiIndex DataFrame identical to yfinance output format
    columns = pd.MultiIndex.from_product([['Adj Close'], ['BND', 'SPY', 'TSLA']], names=['Price', 'Ticker'])
    df = pd.DataFrame(index=date_range, columns=columns)
    
    df[('Adj Close', 'BND')] = bnd_prices
    df[('Adj Close', 'SPY')] = spy_prices
    df[('Adj Close', 'TSLA')] = tsla_prices
    
    # Add a flat dummy column for 'Close' just to satisfy multi-layer indexing
    df.columns.names = [None, 'Ticker']
    return df

if __name__ == "__main__":
    import os
    df = generate_backup_data()
    os.makedirs('data/processed', exist_ok=True)
    # Extract just the asset columns directly as a clean dataframe for processing
    prices = df['Adj Close']
    prices.to_csv('data/processed/cleaned_prices.csv')
    
    # Calculate returns and save
    returns = prices.pct_change().dropna()
    returns.to_csv('data/processed/daily_returns.csv')
    print("Offline backup data created inside data/processed/ successfully!")