import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.title("📈 Stock Market Predictor")


st.sidebar.header("Stock Settings")

stock = st.sidebar.text_input("Enter stock symbol", "GOOG")
start = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("2025-09-15"))

st.sidebar.write("Click below to run:")
run_button = st.sidebar.button("Run Prediction")

if run_button:

    
    st.subheader("🔽 Downloading Data...")

    data = yf.download(stock, start=start, end=end)
    data.reset_index(inplace=True)

    st.write("### Raw Data (Head)")
    st.write(data.head())

    data.ffill(inplace=True)
    data.dropna(inplace=True)

    
    st.subheader("📊 Feature Engineering")

    data['MA_100'] = data['Close'].rolling(window=100).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()

    data["Return"] = data["Close"].pct_change()
    data["Target"] = data["Close"].shift(-1)
    data = data.dropna().reset_index(drop=True)

   
    st.subheader("📉 Close Price with MA100 & MA200")

    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close', linewidth=1)
    plt.plot(data['Date'], data['MA_100'], label='MA 100', linewidth=1.2, linestyle='--')
    plt.plot(data['Date'], data['MA_200'], label='MA 200', linewidth=1.2, linestyle='-.')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{stock} — Close with MA100 & MA200")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig1)

    
    st.subheader(" Model Training")

    features = ["Close", "MA_10", "MA_50", "Return", "Volume"]
    X = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    
    st.subheader("📈 Actual vs Predicted (Test Set)")

    fig2 = plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=1)
    plt.plot(y_test.index
    , y_pred, label='Predicted', linewidth=1, linestyle='--')
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title(f"{stock} — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    r2 = r2_score(y_test, y_pred)

    last_row = data.iloc[-1][features].values.reshape(1, -1)
    predicted_price = model.predict(last_row)[0]

    predicted_int = int(predicted_price)
    low_range = int(predicted_price * 0.98)
    high_range = int(predicted_price * 1.02)

    st.subheader(" Prediction Result")

    st.write(f"### **R² Score:** `{r2:.4f}`")
    st.info(f"Expected integer range: **${low_range} - ${high_range}**")

else:
    st.info("👈 Enter stock details and click **Run Prediction**")
