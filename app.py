import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import time

# Streamlit page configuration
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Aktuellen Bitcoin-Preis von CoinCap API abfragen
def get_current_price():
    url = "https://api.coincap.io/v2/assets/bitcoin"
    try:
        response = requests.get(url)
        data = response.json()

        # Überprüfen der Struktur der Antwort
        if 'data' in data:
            if 'priceUsd' in data['data']:
                return float(data['data']['priceUsd'])
            else:
                st.error(f"Fehler beim Abrufen des aktuellen Preises: Kein 'priceUsd' in den Daten vorhanden.")
                return None
        else:
            st.error(f"Fehler beim Abrufen des aktuellen Preises: {data}")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Historische Bitcoin-Daten für die letzten 30 Minuten abrufen
def get_historical_data():
    url = "https://api.coincap.io/v2/assets/bitcoin/history"
    params = {
        'interval': 'm1',  # Minute als Intervall
        'start': str(int(time.time() - 3600)),  # 1 Stunde zurück
        'end': str(int(time.time()))
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()

        # Überprüfen, ob 'data' vorhanden ist und den Preis korrekt extrahieren
        if 'data' in data:
            historical_prices = []
            for data_point in data['data']:
                if 'priceUsd' in data_point:
                    historical_prices.append(float(data_point['priceUsd']))
                else:
                    st.warning(f"Kein 'priceUsd' für Datenpunkt {data_point}")
            return historical_prices
        else:
            st.error(f"Fehler beim Abrufen der historischen Daten: {data}")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return None

# Random Forest Modell für die Vorhersage der Preise
def predict_price(historical_prices):
    time_steps = np.arange(len(historical_prices))  # Zeitstempel als Features
    X = time_steps.reshape(-1, 1)  # Zeitstempel als Features für das Modell
    y = historical_prices  # Preisdaten als Zielwert

    # Modell trainieren
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Vorhersage für die nächsten 3 Zeitpunkte (1, 5 und 10 Minuten)
    future_times = np.array([[len(historical_prices)], [len(historical_prices) + 5], [len(historical_prices) + 10]])
    predictions = model.predict(future_times)

    return predictions

# Streamlit UI
def app():
    # Titel
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # Aktuellen Preis abrufen
    current_price = get_current_price()
    if current_price:
        st.subheader(f"💰 Aktueller Preis: ${current_price:,.2f}")

    # Historische Daten abrufen
    historical_data = get_historical_data()
    if historical_data:
        # Preisdaten extrahieren
        historical_prices = historical_data

        # RSI berechnen
        rsi = calculate_rsi(historical_prices)
        st.subheader(f"📊 RSI der letzten 30 Minuten")
        st.write(f"Letzter RSI-Wert: {rsi:.2f}")

        # Vorhersage berechnen
        predictions = predict_price(historical_prices)
        st.subheader(f"📉 Preisvorhersage")
        st.write(f"Vorhergesagter Preis in 1 Minute: ${predictions[0]:,.2f}")
        st.write(f"Vorhergesagter Preis in 5 Minuten: ${predictions[1]:,.2f}")
        st.write(f"Vorhergesagter Preis in 10 Minuten: ${predictions[2]:,.2f}")

# RSI-Berechnung
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None  # Nicht genug Daten für RSI-Berechnung
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])

    if avg_loss == 0:
        return 100  # Verhindert Division durch Null

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Automatische Aktualisierung alle 60 Sekunden
if __name__ == "__main__":
    app()

