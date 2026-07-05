# GMF Investments: Time Series Forecasting & Portfolio Optimization

An institutional-grade data engineering, predictive modeling, and analytics pipeline built for **Guide Me in Finance (GMF) Investments**. This repository leverages time-series forecasting frameworks and deep learning paradigms to model market trends, mitigate risk exposure, and optimize asset allocation structures for personalized wealth management.

---

## 📂 Project Structure

```text
portfolio-optimization/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── processed/                  # Transformed, stationary data assets (CSV)
├── models/
│   └── arima_model.pkl             # Serialized classical forecasting weights
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb   # Task 1: Pipeline & Visual Exploratory Engine
│   └── 02_model_development.ipynb        # Task 2: Predictive Forecasting & Verification
├── src/
│   └── __init__.py
├── scripts/
│   ├── __init__.py
│   ├── data_pipeline.py            # Preprocessing, data scaling, & ADF tests
│   ├── generate_offline_data.py    # Fail-safe offline synthetic baseline data generator
│   └── forecast_models.py          # Classical & sequential deep-learning engines
└── tests/
    └── __init__.py