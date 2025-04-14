import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# URL für den aktuellen Preis und historische Daten von CoinGecko
current_price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
historical_data_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

# Funktion zum Abrufen des aktuellen Preises
def get_current_price():
    try:
        response = requests.get(current_price_url)
        data = response.json()
        
        # Prüfen, ob 'bitcoin' und 'usd' vorhanden sind
        if "bitcoin" in data and "usd" in data["bitcoin"]:
            current_price = data["bitcoin"]["usd"]
            return current_price
        else:
            st.error(f"Fehler beim Abrufen des aktuellen Preises: {data}")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Funktion zum Abrufen historischer Daten (minütlich)
def get_historical_data():
    try:
        params = {
            "vs_currency": "usd",
            "days": "1",  # 1 Tag
            "interval": "minute",  # Gültiger Intervallwert
        }
        response = requests.get(historical_data_url, params=params)
        data = response.json()

        # Prüfen, ob 'prices' in den Daten vorhanden sind
        if "prices" in data:
            historical_prices = data["prices"]
            return pd.DataFrame(historical_prices, columns=["timestamp", "price"])
        else:
            st.error(f"Fehler beim Abrufen der historischen Daten: {data}")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return None

# Funktion zum Berechnen des RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit-App
def app():
    st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # Aktuellen Preis abrufen
    current_price = get_current_price()
    if current_price is not None:
        st.subheader("💰 Aktueller Preis")
        st.write(f"${current_price:,.2f}")
    
    # Historische Daten abrufen
    historical_data = get_historical_data()
    if historical_data is not None:
        # Zeitstempel in Datetime umwandeln und als Index setzen
        historical_data["timestamp"] = pd.to_datetime(historical_data["timestamp"], unit="ms")
        historical_data.set_index("timestamp", inplace=True)

        # Berechnung des RSI
        rsi = calculate_rsi(historical_data["price"])
        st.subheader("📊 RSI der letzten 30 Minuten")
        st.write(f"Letzter RSI-Wert: {rsi.iloc[-1]:.2f}")

        # Vorhersage der nächsten 1, 5 und 10 Minuten
        # Hier wird ein einfaches Random-Modell verwendet, um eine Vorhersage zu simulieren
        st.subheader("📉 Preisvorhersage")

        # Vorhersage für 1, 5, 10 Minuten
        prediction_1min = current_price + np.random.uniform(-100, 100)
        prediction_5min = current_price + np.random.uniform(-200, 200)
        prediction_10min = current_price + np.random.uniform(-300, 300)

        st.write(f"Vorhergesagter Preis in 1 Minute: ${prediction_1min:,.2f}")
        st.write(f"Vorhergesagter Preis in 5 Minuten: ${prediction_5min:,.2f}")
        st.write(f"Vorhergesagter Preis in 10 Minuten: ${prediction_10min:,.2f}")

    # Verzögerung, um das API-Limit nicht zu überschreiten
    time.sleep(60)  # 60 Sekunden Verzögerung
    st.experimental_rerun()

if __name__ == "__main__":
    app()

