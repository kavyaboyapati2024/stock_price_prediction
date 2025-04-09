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
from stocks import run_stock_prediction_analysis
import pandas as pd
import datetime



# Page setup
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>📊 Stock Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Ticker selection
st.markdown("### 🔍 Select a Stock Ticker")
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

selected_ticker_name = st.selectbox("📌 Choose a Stock", list(famous_tickers.keys()))
if famous_tickers[selected_ticker_name] == "OTHER":
    ticker = st.text_input("✏ Enter Ticker Symbol", "")
else:
    ticker = famous_tickers[selected_ticker_name]

# Date input section
st.markdown("### 🗓 Select Date Range")
today = datetime.date.today()
default_start = today - datetime.timedelta(days=730)
default_end = today

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=default_start)
with col2:
    end_date = st.date_input("End Date", value=default_end)

# Prediction button

run_button = st.button("🚀 Predict Price")

if run_button:
    if not ticker:
        st.warning("❗ Please enter a valid stock ticker symbol.")
    elif (end_date - start_date).days < 365:
        st.warning("⚠ Please select at least a one-year range.")
    else:
        with st.spinner("⏳ Running analysis..."):
            try:
                results = run_stock_prediction_analysis(ticker, str(start_date), str(end_date))
                next_day = pd.to_datetime(end_date) + pd.tseries.offsets.BDay(1)
                st.success(f"✅ Prediction complete for: *{next_day.date()}*")
                st.markdown("---")

                # Prediction Summary
                summary_df = results["summary_df"]
                linear_summary = summary_df[summary_df["Model"] == "Linear Regression"]

                

                if linear_summary.empty:
                    st.warning("⚠ No prediction available.")
                else:
                    predicted_price = float(linear_summary["Predicted Close"].values[0])

                try:
                    if ("Actual Close" in linear_summary.columns and not linear_summary.empty):
                        actual_value = linear_summary["Actual Close"].values[0]
                        if pd.notna(actual_value) and actual_value != "N/A":
                            actual_price = float(actual_value)
                        else:
                            actual_price = None
                    else:
                        actual_price = None
                except Exception as e:
                    actual_price = None  



                    
                   
                    
               # Display predicted and actual prices
                if actual_price is not None:
                    st.markdown(f"""
                        <div style='background-color: #F0F8FF; padding: 30px 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                            <h2 style='color: #1B4F72;'>📌 Predicted vs Actual Closing Price</h2>
                            <p style='font-size: 36px; font-weight: bold; color: #2E86C1;'>Predicted: ${predicted_price:.2f}</p>
                            <p style='font-size: 36px; font-weight: bold; color: #117A65;'>Actual: ${actual_price:.2f}</p>
                            <p style='font-size: 16px; color: #7D7D7D;'>for trading day {next_day.date()}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background-color: #F0F8FF; padding: 30px 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                            <h2 style='color: #1B4F72;'>📌 Predicted Closing Price</h2>
                            <p style='font-size: 48px; font-weight: bold; color: #2E86C1;'>${predicted_price:.2f}</p>
                            <p style='font-size: 16px; color: #7D7D7D;'>for trading day {next_day.date()}</p>
                        </div>
                    """, unsafe_allow_html=True)



                # Plot Section
                st.subheader("📉 Prediction Plot")
                if "Linear Regression" in results["plots"]:
                    st.image(results["plots"]["Linear Regression"], use_container_width=True)
                else:
                    st.warning("⚠ Plot not available.")

            except ValueError as e:
                st.error(f"❌ {str(e)}")

