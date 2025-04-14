import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# === Konfiguration ===
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

API_KEY_TWELVE = "1c83ee150f8344eaa397d1d90a9da4f4"

# === Funktionen ===
def get_current_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

def get_historical_data():
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "BTC/USD",
        "interval": "1min",
        "outputsize": 30,
        "apikey": API_KEY_TWELVE
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "values" in data:
            return list(reversed(data["values"]))  # Chronologisch sortieren
        else:
            st.warning(f"Twelve Data Fehler: {data.get('message', 'Keine Daten erhalten')}")
            return []
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return []

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi = []
    for i in range(period, len(prices)):
        if avg_loss == 0:
            rs = 0
        else:
            rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))

        if i + 1 < len(prices):
            delta = deltas[i]
            gain = max(delta, 0)
            loss = max(-delta, 0)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

    return rsi[-1] if rsi else None

def predict_price(current_price):
    # Einfaches Random-Modell (Demo)
    return {
        "1min": current_price * (1 + np.random.normal(0, 0.0005)),
        "5min": current_price * (1 + np.random.normal(0, 0.0015)),
        "10min": current_price * (1 + np.random.normal(0, 0.0025))
    }

# === App-Logik ===
def app():
    st.subheader("💰 Aktueller Preis:")
    price = get_current_price()
    if price:
        st.metric("Bitcoin Preis (USD)", f"${price:,.2f}")

    st.subheader("📊 RSI der letzten 30 Minuten")
    historical_data = get_historical_data()

    if historical_data:
        prices = [float(entry['close']) for entry in historical_data]
        rsi = calculate_rsi(prices)
        if rsi:
            st.write(f"Letzter RSI-Wert: **{rsi:.2f}**")
        else:
            st.write("Nicht genügend Daten zur Berechnung des RSI.")

        st.subheader("📉 Preisvorhersage")
        if price:
            prediction = predict_price(price)
            st.write
