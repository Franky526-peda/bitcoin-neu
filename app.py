import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Seite konfigurieren
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# API-Konfiguration
API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

# RSI-Berechnung ohne externe Bibliothek
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Daten abrufen mit Fehlerbehandlung
def fetch_btc_data():
    try:
        params = {
            "vs_currency": "usd",
            "days": "1",
            "interval": "minutely"
        }
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Fehler beim Abrufen der Daten: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Bitcoin-Daten: {e}")
        return None

# Vorhersagefunktion
def predict_future(df, minutes):
    if len(df) < 10:
        return "Nicht genügend Daten"
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    X = df[['minutes']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    future_minute = df['minutes'].max() + minutes
    prediction = model.predict([[future_minute]])[0]
    return round(prediction, 2)

# Hauptfunktion
def main():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")

    data = fetch_btc_data()
    if not data or 'prices' not in data:
        st.error("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return

    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['price'].astype(float)

    # Technische Indikatoren berechnen
    df['SMA_10'] = df['price'].rolling(window=10).mean()
    df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['RSI'] = compute_rsi(df['price'])

    aktueller_preis = df['price'].iloc[-1]
    st.subheader(f"Aktueller Preis: ${aktueller_preis:,.2f}")

    pred_1 = predict_future(df, 1)
    pred_5 = predict_future(df, 5)
    pred_10 = predict_future(df, 10)

    st.write(f"📉 Vorhersage in 1 Minute: {pred_1 if isinstance(pred_1, str) else f'${pred_1:,.2f}'}")
    st.write(f"⏱️ Vorhersage in 5 Minuten: {pred_5 if isinstance(pred_5, str) else f'${pred_5:,.2f}'}")
    st.write(f"⏳ Vorhersage in 10 Minuten: {pred_10 if isinstance(pred_10, str) else f'${pred_10:,.2f}'}")

    st.subheader("📊 Technische Indikatoren:")
    last = df.iloc[-1]
    st.write(f"SMA (10): {last['SMA_10']:.2f}")
    st.write(f"EMA (10): {last['EMA_10']:.2f}")
    st.write(f"RSI: {last['RSI']:.2f}")

    st.line_chart(df.set_index('timestamp')[['price', 'SMA_10', 'EMA_10']].dropna())

if __name__ == "__main__":
    main()

