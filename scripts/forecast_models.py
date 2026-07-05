import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import pickle

def split_time_series_data(df, target_col='TSLA', split_date='2025-01-01'):
    """
    Chronologically splits data to preserve temporal order.
    """
    train = df[df.index < split_date][target_col]
    test = df[df.index >= split_date][target_col]
    return train, test

def run_arima_forecast(train, test, order=(1, 1, 1)):
    """
    Fits an ARIMA model and generates out-of-sample forecasts dynamically.
    """
    print(f"Fitting ARIMA{order} Model...")
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    # Forecast for the duration of the test set length
    forecast = fitted_model.forecast(steps=len(test))
    forecast.index = test.index
    return forecast, fitted_model

def create_lstm_sequences(data, window_size=60):
    """
    Converts 1D array into sliding sequence windows for LSTM tracking.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def calculate_evaluation_metrics(y_true, y_pred):
    """
    Computes professional model accuracy metrics: MAE, RMSE, MAPE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4)
    }