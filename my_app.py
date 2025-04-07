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
import streamlit as st
from mlsp import run_stock_prediction_analysis  # make sure this matches your file name
import pandas as pd

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# Famous tickers dropdown
famous_tickers = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Meta Platforms Inc. (META)": "META",
    "Tesla Inc. (TSLA)": "TSLA",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Netflix Inc. (NFLX)": "NFLX",
    "JPMorgan Chase & Co. (JPM)": "JPM",
    "Other (Enter Manually)": "OTHER"
}

selected_ticker_name = st.selectbox("Choose a Stock Ticker", list(famous_tickers.keys()))
if famous_tickers[selected_ticker_name] == "OTHER":
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT)", "")
else:
    ticker = famous_tickers[selected_ticker_name]

# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

# Run Analysis
if st.button("Run Prediction"):
    if not ticker:
        st.warning("Please enter a valid stock ticker symbol.")
    elif (end_date - start_date).days < 365:
        st.warning("⚠️ Please make sure to have at least one year difference between start date and end date for better prediction.")
    else:
        with st.spinner("Running analysis..."):
            results = run_stock_prediction_analysis(ticker, str(start_date), str(end_date))

            # Summary Table
            st.subheader("📊 Prediction Summary")
            next_day = pd.to_datetime(end_date) + pd.tseries.offsets.BDay(1)
            st.markdown(f"🗓️ **Predictions are for the trading day:** `{next_day.date()}`")
            st.dataframe(results["summary_df"], use_container_width=True)

            # Evaluation Metrics
            st.subheader("📈 Model Evaluation Metrics")
            st.dataframe(results["metrics_df"], use_container_width=True)

            # Plots
            st.subheader("📉 Model Performance Plots")
            for model, path in results["plots"].items():
                st.markdown(f"**{model}**")
                st.image(path, use_container_width=True)

