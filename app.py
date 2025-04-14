import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression

# Dein Twelve Data API-Schlüssel
API_KEY = "1c83ee150f8344eaa397d1d90a9da4f4"
BASE_URL = "https://api.twelvedata.com"

# RSI-Berechnung
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_val, down_val = delta, 0.
        else:
            up_val, down_val = 0., -delta

        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# Aktuellen Preis abrufen
def get_current_price():
    url = f"{BASE_URL}/price?symbol=BTC/USD&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return float(data["price"]) if "price" in data else None

# Historische Minutenpreise abrufen
def get_historical_prices():
    interval = "1min"
    outputsize = 60
    url = f"{BASE_URL}/time_series?symbol=BTC/USD&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "values" in data:
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)
        df = df.sort_values("datetime")
        return df
    return None

# Lineares Regressionsmodell
def predict_prices(df, future_minutes):
    df["timestamp"] = df["datetime"].astype(np.int64) // 10**9
    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    last_timestamp = X[-1][0]
    future_predictions = {}
    for minutes in future_minutes:
        future_time = last_timestamp + minutes * 60
        future_pred = model.predict([[future_time]])[0]
        future_predictions[minutes] = round(future_pred, 2)

    return future_predictions

# Haupt-App
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    current_price = get_current_price()
    if current_price:
        st.subheader("💰 Aktueller Preis:")
        st.markdown(f"**Bitcoin Preis (USD)**\n\n${current_price:,.2f}")
    else:
        st.error("❌ Konnte aktuellen Preis nicht abrufen.")
        return

    df = get_historical_prices()
    if df is not None and len(df) > 30:
        closes = df["close"].values
        rsi = calculate_rsi(closes, period=14)
        st.subheader("📊 RSI der letzten 30 Minuten")
        st.write(f"Letzter RSI-Wert: {rsi[-1]:.2f}")

        st.subheader("📉 Preisvorhersage mit linearer Regression")
        predictions = predict_prices(df, [1, 5, 10])
        for mins, pred in predictions.items():
            st.write(f"Vorhergesagter Preis in {mins} Minute(n): ${pred:,.2f}")
    else:
        st.warning("Keine historischen Preisdaten verfügbar – keine Vorhersage möglich.")

if __name__ == "__main__":
    app()

    # Countdown zur nächsten Aktualisierung
    countdown = st.empty()
    for i in range(60, 0, -1):
        countdown.info(f"🔄 Aktualisierung in {i} Sekunden ...")
        time.sleep(1)
    st.experimental_rerun()
