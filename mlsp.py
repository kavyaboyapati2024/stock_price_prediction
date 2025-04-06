




# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

def run_stock_prediction_analysis(ticker, start_date, end_date_input):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import warnings
    import json
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    import os

    warnings.filterwarnings('ignore')

    ticker = ticker.upper()
    end_date = (pd.to_datetime(end_date_input) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Load data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} in the given date range.")


    df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Technical Indicators
    df['SMA_14'] = SMAIndicator(df['Close'], window=14).sma_indicator()
    df['EMA_14'] = EMAIndicator(df['Close'], window=14).ema_indicator()
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Lag features & Target
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(inplace=True)  # keep this if you need "Date" as a column


    features = ['Open', 'High', 'Low', 'Volume', 'SMA_14', 'EMA_14',
                'RSI_14', 'MACD', 'MACD_Signal', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train Models
    model_XGB = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                             subsample=0.9, colsample_bytree=0.8, gamma=0,
                             reg_alpha=0.5, reg_lambda=1, random_state=42,
                             objective='reg:squarederror')
    model_XGB.fit(X_train, y_train)
    y_pred_XGB = model_XGB.predict(X_test)

    model_GB = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05,
                                         max_depth=4, subsample=0.8,
                                         min_samples_split=3, min_samples_leaf=2,
                                         random_state=42)
    model_GB.fit(X_train, y_train)
    y_pred_GB = model_GB.predict(X_test)

    model_LR = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)

    model_RF = RandomForestRegressor(n_estimators=150, max_depth=12,
                                     max_features='sqrt', min_samples_split=4,
                                     min_samples_leaf=2, bootstrap=True, random_state=42)
    model_RF.fit(X_train, y_train)
    y_pred_RF = model_RF.predict(X_test)

    def evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < 0.02) * 100
        return {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "accuracy_within_2pct": round(accuracy, 2)
        }

    metrics = {
        "XGBoost": evaluate_model("XGBoost", model_XGB, X_test, y_test),
        "Gradient Boosting": evaluate_model("Gradient Boosting", model_GB, X_test, y_test),
        "Linear Regression": evaluate_model("Linear Regression", model_LR, X_test, y_test),
        "Random Forest": evaluate_model("Random Forest", model_RF, X_test, y_test),
    }

    # Predict next trading day
    last_row = df[features].iloc[-1:]
    last_row_scaled = scaler.transform(last_row)
    pred_xgb = model_XGB.predict(last_row_scaled)[0]
    pred_gb = model_GB.predict(last_row_scaled)[0]
    pred_lr = model_LR.predict(last_row_scaled)[0]
    pred_rf = model_RF.predict(last_row_scaled)[0]

    def get_next_trading_day(date, ticker):
        date = pd.to_datetime(date)
        next_window = yf.download(ticker, start=date + pd.Timedelta(days=1), end=date + pd.Timedelta(days=7))
        if not next_window.empty:
            return next_window.index[0], next_window['Close'].iloc[0]
        return None, None

    next_trading_day, actual_next_day_price = get_next_trading_day(end_date_input, ticker)

    actual_close_value = float(actual_next_day_price) if actual_next_day_price is not None else "N/A"

    summary = {
        "next_trading_day": str(next_trading_day.date()) if next_trading_day else "N/A",
        "actual_next_close": actual_close_value,
        "predictions": {
            "XGBoost": round(pred_xgb, 2),
            "Gradient Boosting": round(pred_gb, 2),
            "Linear Regression": round(pred_lr, 2),
            "Random Forest": round(pred_rf, 2),
        }
    }

   

    def save_plot(y_test, y_pred, model_name, filename, dates):
        plt.figure(figsize=(10, 6))
        plt.plot(dates, y_test, label="Actual", color='blue')
        plt.plot(dates, y_pred, label=f"Predicted ({model_name})", color='red')
        plt.title(f"Actual vs Predicted: {model_name}")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    # Save plots
    
   # Extract dates for test set
    test_dates = df['Date'].iloc[-len(y_test):]

    # Call with dates
    save_plot(y_test, y_pred_XGB, "XGBoost", "xgb_plot.png", test_dates)
    save_plot(y_test, y_pred_GB, "Gradient Boosting", "gb_plot.png", test_dates)
    save_plot(y_test, y_pred_LR, "Linear Regression", "lr_plot.png", test_dates)
    save_plot(y_test, y_pred_RF, "Random Forest", "rf_plot.png", test_dates)

    plot_files = {
        "XGBoost": "xgb_plot.png",
        "Gradient Boosting": "gb_plot.png",
        "Linear Regression": "lr_plot.png",
        "Random Forest": "rf_plot.png"
    }

    # Format Summary as DataFrame
    summary_df = pd.DataFrame([
        {"Model": model, "Predicted Close": pred, "Actual Close": summary["actual_next_close"]}
        for model, pred in summary["predictions"].items()
    ])
    summary_df.index.name = 'Index'

    # Format Metrics as DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.name = "Model"
    metrics_df.reset_index(inplace=True)
    
    return {
        "metrics": metrics,
        "summary": summary,
        "plots": plot_files,
        "y_test": y_test,
        "predictions": {
            "XGBoost": y_pred_XGB,
            "Gradient Boosting": y_pred_GB,
            "Linear Regression": y_pred_LR,
            "Random Forest": y_pred_RF
        },
        "summary_df": summary_df,
        "metrics_df": metrics_df
    }
   
print("success")
