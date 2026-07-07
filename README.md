Markdown# GMF Investments: Predictive Stock Forecasting & Modern Portfolio Theory (MPT) Optimization

An end-to-end quantitative financial engineering pipeline that forecasts stock trends using time-series analysis and constructs mathematically optimized portfolios to maximize risk-adjusted returns. This project uses historical asset data for Tesla (**TSLA**), SPDR S&P 500 ETF (**SPY**), and Vanguard Total Bond Market ETF (**BND**).

---

## 📂 Project Architecture

```text
portfolio-optimization/
│
├── data/
│   └── processed/
│       ├── cleaned_prices.csv
│       ├── daily_returns.csv
│       └── tsla_future_forecast.csv
│
├── models/
│   └── arima_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_future_forecasting.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_strategy_backtesting.ipynb
│
├── scripts/
│   └── forecast_models.py
└── README.md
Pipeline Tasks BreakdownTask 1: Data Preprocessing & Exploratory Data Analysis (EDA)Objective: Clean raw price distributions, check for structural stationarity, and calculate log/simple relative daily returns.Key Findings: Identified low correlation boundaries between fixed-income instruments ($BND$) and volatile equities ($TSLA$), establishing a mathematical foundation for diversification.Task 2: Time-Series Model DevelopmentObjective: Train and validate predictive statistical pipelines (ARIMA/SARIMA) on historical sequences.Metrics Tracked: Evaluated performance utilizing Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics to isolate the most robust configurations.Task 3: Out-of-Sample Future ForecastingObjective: Project Tesla's price path 6–12 months into the future alongside expanding 95% confidence risk funnels.Takeaway: Explored how uncertainty expands non-linearly across distant horizons due to variance compounding, demonstrating the structural limitations of long-range point forecasts.Task 4: Modern Portfolio Theory (MPT) OptimizationObjective: Assemble an optimal portfolio mapping out an Efficient Frontier using annualized expected return vectors and asset covariances.Execution Engine: Uses a high-performance grid search simulation to locate the Maximum Sharpe Ratio Portfolio and Minimum Volatility Portfolio without environment-level locks.Task 5: Strategy BacktestingObjective: Validate the optimized asset allocations out-of-sample against a traditional static passive benchmark (60% SPY / 40% BND).Metrics Computed: Evaluated Total Return, Annualized Sharpe Ratio, and Maximum Drawdowns to verify empirical viability under active market scenarios.
 Execution Guide1. Environment SetupEnsure your environment contains standard scientific data science packages:Bashpip install pandas numpy matplotlib seaborn scipy
2. Running the SystemExecute each module sequentially via the Jupyter notebooks located inside the notebooks/ directory:Run 01_data_preprocessing_eda.ipynb to clean raw assets.Open 02_model_development.ipynb to optimize and save the time-series forecasting model.Use 03_future_forecasting.ipynb to generate forward-looking prices.Execute 04_portfolio_optimization.ipynb to compute weights along the Efficient Frontier.Finish with 05_strategy_backtesting.ipynb to generate historical performance curves.
Core Performance MetricsThe system outputs institutional performance analytics summarizing both risk and return:Strategy Allocation FrameworkExpected Annual ReturnAnnualized VolatilitySharpe RatioMax DrawdownOptimized Strategy Portfolio[Dynamic Variable][Dynamic Variable][Dynamic][Dynamic]Passive Benchmark (60/40)[Dynamic Variable][Dynamic Variable][Dynamic][Dynamic]
 Strategic DisclaimersFriction Free Simulation: This simulation does not explicitly bake in capital gains tax, clearing fees, or platform slippage.Risk Parameters: Past outperformance metrics are never a definitive guarantee of future live market profits.