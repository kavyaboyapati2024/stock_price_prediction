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

# +
def run_stock_prediction_analysis(ticker, start_date, end_date_input):
    import yfinance as yf
    import pandas as pd
    import numpy as np
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
    import warnings

    warnings.filterwarnings('ignore')

    ticker = ticker.upper()
    end_date = (pd.to_datetime(end_date_input) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Load data
    df = yf.download(ticker, start=start_date, end=end_date)

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
    df.reset_index(inplace=True)

    features = ['Open', 'High', 'Low', 'Volume', 'SMA_14', 'EMA_14',
                'RSI_14', 'MACD', 'MACD_Signal', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model_LR = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)
    
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
       
        "Linear Regression": evaluate_model("Linear Regression", model_LR, X_test, y_test),
       
    }

    # Predict next trading day
    last_row = df[features].iloc[-1:]
    last_row_scaled = scaler.transform(last_row)
   
    pred_lr = model_LR.predict(last_row_scaled)[0]
   

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
            
            "Linear Regression": round(pred_lr, 2),
           
        }
    }
    summary_df = pd.DataFrame([{
    "Model": "Linear Regression",
    "Predicted Close": round(pred_lr, 2),
    "Actual Close": actual_close_value,
    "Date": str(next_trading_day.date()) if next_trading_day else "N/A"
}])


    def save_plot(y_test, y_pred, model_name, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test.values, label="Actual", color='blue')
        plt.plot(y_test.index, y_pred, label=f"Predicted ({model_name})", color='red')
        plt.title(f"Actual vs Predicted: {model_name}")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Save plots
    os.makedirs("plots", exist_ok=True)
    plot_files = {}
   
    save_plot(y_test, y_pred_LR, "Linear Regression", "plots/lr_plot.png")

    plot_files = {
        "Linear Regression": "plots/lr_plot.png",
    }

    return {
        "metrics": metrics,
        "summary_df": summary_df,
        "plots": plot_files,
        "y_test": y_test,
        "predictions": {
            "Linear Regression": y_pred_LR,
        }
    }


print("done")
