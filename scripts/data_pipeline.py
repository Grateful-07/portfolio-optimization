import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def fetch_financial_data(tickers=["TSLA", "BND", "SPY"], start_date="2015-01-01", end_date="2026-06-30"):
    """
    Fetches historical daily market data from YFinance.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def preprocess_data(df):
    """
    Cleans data, extracts Adjusted Close prices, and handles missing data via forward/backward filling.
    """
    # Isolate Adjusted Close
    adj_close = df['Adj Close'].copy()
    
    # Check for missing values
    missing_counts = adj_close.isnull().sum()
    print("Missing values per asset before cleaning:\n", missing_counts)
    
    # Handle missing values using forward fill then backward fill (handles holidays/weekends smoothly)
    adj_close = adj_close.ffill().bfill()
    
    return adj_close

def calculate_metrics(df):
    """
    Calculates Daily Returns and Rolling Volatility (21-day window).
    """
    returns = df.pct_change().dropna()
    rolling_vol = returns.rolling(window=21).std() * np.sqrt(252) # Annualized 21-day rolling volatility
    return returns, rolling_vol

def check_stationarity(series, asset_name, metric_name):
    """
    Performs the Augmented Dickey-Fuller (ADF) test.
    """
    result = adfuller(series)
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    summary = {
        "Asset": asset_name,
        "Metric": metric_name,
        "ADF Statistic": round(result[0], 4),
        "p-value": round(p_value, 6),
        "Stationary": is_stationary
    }
    return summary

def calculate_risk_metrics(returns, confidence_level=0.95):
    """
    Calculates Value at Risk (VaR) and annualized Sharpe Ratio (assuming Risk-Free Rate = 0).
    """
    risk_summary = {}
    for col in returns.columns:
        asset_returns = returns[col]
        
        # Historical VaR
        var_limit = np.percentile(asset_returns, (1 - confidence_level) * 100)
        
        # Sharpe Ratio (Annualized)
        mean_ret = asset_returns.mean()
        std_dev = asset_returns.std()
        sharpe = (mean_ret / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        
        risk_summary[col] = {
            "Historical VaR (5%)": round(var_limit, 4),
            "Annualized Sharpe Ratio": round(sharpe, 4)
        }
    return pd.DataFrame(risk_summary).T