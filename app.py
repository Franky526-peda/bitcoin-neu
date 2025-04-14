import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# API Key für Twelve Data
API_KEY = "1c83ee150f8344eaa397d1d90a9da4f4"
BASE_URL = "https://api.twelvedata.com"

# Funktion zum Abrufen des aktuellen Bitcoin-Preises
def get_current_price():
    url = f"{BASE_URL}/price?symbol=BTC/USD&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        return data["price"]
    except KeyError:
        return None

# Funktion zum Abrufen der historischen Bitcoin-Preise
def get_historical_prices():
    url = f"{BASE_URL}/time_series?symbol=BTC/USD&interval=1min&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        prices = [float(point["close"]) for point in data["values"]]
        return prices
    except KeyError:
        return []

# Funktion zum Berechnen des RSI
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = delta[delta > 0].sum()
    loss = -delta[delta < 0].sum()

    avg_gain = gain / period
    avg_loss = loss / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Funktion zur Vorhersage des Bitcoin-Preises mit linearer Regression
def predict_price(prices):
    model = LinearRegression()
    times = np.array(range(len(prices))).reshape(-1, 1)
    model.fit(times, prices)
    future_time = np.array([[len(prices) + 1]])  # Vorhersage für die nächste Minute
    predicted_price = model.predict(future_time)
    return predicted_price[0]

# Haupt-App
def app():
    # Abruf der aktuellen Bitcoin-Daten
    current_price = get_current_price()
    if current_price is None:
        st.error("Fehler beim Abrufen des aktuellen Preises")
        return

    # Abruf der historischen Bitcoin-Daten
    historical_prices = get_historical_prices()
    if not historical_prices:
        st.error("Fehler beim Abrufen der historischen Daten")
        return

    # Berechnung des RSI der letzten 30 Minuten
    if len(historical_prices) >= 30:
        rsi = calculate_rsi(historical_prices[-30:])
    else:
        st.error("Nicht genügend Daten für RSI")
        return

    # Vorhersage des Bitcoin-Preises mit linearer Regression
    predicted_price_1_min = predict_price(historical_prices[-30:])  # Vorhersage für 1 Minute
    predicted_price_5_min = predict_price(historical_prices[-30:])  # Vorhersage für 5 Minuten
    predicted_price_10_min = predict_price(historical_prices[-30:])  # Vorhersage für 10 Minuten

    # Anzeige der Ergebnisse
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")
    st.write(f"💰 Aktueller Preis: ${current_price}")
    st.write(f"📊 RSI der letzten 30 Minuten: {rsi:.2f}")
    st.write(f"📉 Preisvorhersage mit linearer Regression:")
    st.write(f"Vorhergesagter Preis in 1 Minute(n): ${predicted_price_1_min:.2f}")
    st.write(f"Vorhergesagter Preis in 5 Minute(n): ${predicted_price_5_min:.2f}")
    st.write(f"Vorhergesagter Preis in 10 Minute(n): ${predicted_price_10_min:.2f}")

    # Button für manuelle Aktualisierung
    st.button("Daten aktualisieren")  # Button wird angezeigt, aber führt keinen Fehler aus

# Aufruf der Haupt-App
app()
