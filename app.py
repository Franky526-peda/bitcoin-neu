import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Dein API-Key
API_KEY = "1c83ee150f8344eaa397d1d90a9da4f4"

# Funktionen zum Abrufen der Daten
def get_current_price():
    url = f"https://api.twelvedata.com/price?symbol=BTC/USD&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

def get_historical_data():
    interval = "1min"
    outputsize = 30  # letzte 30 Minuten
    url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["close"] = df["close"].astype(float)
            df = df.sort_values("datetime")
            return df
        else:
            st.error("Keine historischen Daten verfügbar.")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_price(current_price):
    # Einfaches Random-Modell zur Simulation
    predictions = {
        "1 Minute": round(current_price * (1 + np.random.normal(0, 0.002)), 2),
        "5 Minuten": round(current_price * (1 + np.random.normal(0, 0.005)), 2),
        "10 Minuten": round(current_price * (1 + np.random.normal(0, 0.01)), 2),
    }
    return predictions

# Streamlit App
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # Aktueller Preis
    current_price = get_current_price()
    if current_price:
        st.subheader("💰 Aktueller Preis:")
        st.write(f"${current_price:,.2f}")

    # Historische Daten
    df = get_historical_data()
    if df is not None and len(df) >= 15:
        st.subheader("📊 RSI der letzten 30 Minuten")
        rsi_series = calculate_rsi(df["close"])
        latest_rsi = rsi_series.iloc[-1]
        st.write(f"Letzter RSI-Wert: **{latest_rsi:.2f}**")

        # Vorhersagen
        st.subheader("📉 Preisvorhersage")
        predictions = predict_price(current_price)
        for timeframe, price in predictions.items():
            st.write(f"Vorhergesagter Preis in {timeframe}: **${price:,.2f}**")
    else:
        st.warning("Keine historischen Preisdaten verfügbar – keine Vorhersage möglich.")

    st.caption("🔄 Aktualisiert sich alle 60 Sekunden ...")

# App starten
app()

# Automatisch alle 60 Sekunden neuladen
time.sleep(60)
st.experimental_rerun()
