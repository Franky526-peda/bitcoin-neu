import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import random

# CoinMarketCap API Key und Base URL
API_KEY = 'f8d360d4-146c-4f00-8a05-b2f156224c2a'
BASE_URL = "https://pro-api.coinmarketcap.com/v1"

# Funktionen zum Abrufen der Daten
def get_current_price():
    url = f"{BASE_URL}/cryptocurrency/listings/latest"
    params = {
        "start": 1,
        "limit": 1,
        "convert": "USD",
    }
    headers = {
        'X-CMC_PRO_API_KEY': API_KEY,
        'Accept': 'application/json',
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if 'data' in data:
        return data['data'][0]['quote']['USD']['price']
    else:
        st.error("Fehler beim Abrufen des aktuellen Preises")
        return None

def get_historical_data():
    url = f"{BASE_URL}/cryptocurrency/ohlcv/historical"
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_end": int(time.time()),  # Aktueller Zeitpunkt
        "time_start": int(time.time()) - 1800,  # 30 Minuten vorher
        "interval": "minute"
    }
    headers = {
        'X-CMC_PRO_API_KEY': API_KEY,
        'Accept': 'application/json',
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if 'data' in data:
        return data['data']['quotes']
    else:
        st.error("Fehler beim Abrufen der historischen Daten")
        return None

# RSI Berechnung
def calculate_rsi(prices, window=14):
    delta = np.diff(prices)
    gain = delta[delta > 0].sum() / window
    loss = -delta[delta < 0].sum() / window
    
    if loss == 0:
        return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Random Preisvorhersage (Beispiel)
def predict_price():
    return random.uniform(80000, 90000)

# Streamlit App
def app():
    st.title("Bitcoin Predictor – Live Vorhersagen mit RSI")
    
    # Aktuellen Bitcoin Preis abrufen
    current_price = get_current_price()
    if current_price is not None:
        st.subheader(f"💰 Aktueller Preis: ${current_price:,.2f}")

    # Historische Daten abrufen
    historical_data = get_historical_data()
    if historical_data:
        # Preise extrahieren
        prices = [item['quote']['USD']['close'] for item in historical_data]
        
        # RSI berechnen
        if len(prices) >= 14:  # Wir brauchen mindestens 14 Datenpunkte für den RSI
            rsi = calculate_rsi(prices)
            st.subheader(f"📊 RSI der letzten 30 Minuten")
            st.write(f"Letzter RSI-Wert: {rsi:.2f}")
        else:
            st.warning("Nicht genügend Daten für RSI oder Vorhersage verfügbar.")
        
        # Vorhersage erstellen
        st.subheader("📉 Preisvorhersage")
        st.write(f"Vorhergesagter Preis in 1 Minute: ${predict_price():,.2f}")
        st.write(f"Vorhergesagter Preis in 5 Minuten: ${predict_price():,.2f}")
        st.write(f"Vorhergesagter Preis in 10 Minuten: ${predict_price():,.2f}")
    
    # Automatisch alle 60 Sekunden aktualisieren
    st.experimental_rerun()

# Streamlit App ausführen
if __name__ == "__main__":
    app()
