import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# RSI-Berechnung ohne ta
def calculate_rsi(data, period=14):
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# API-Daten abrufen
def get_historical_btc_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '7',  # letzte 7 Tage
        'interval': 'daily'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'prices' in data:
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        else:
            st.error("Fehler in den API-Daten: Die API-Antwort enthält keinen 'prices'-Schlüssel.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen der Bitcoin-Daten: {e}")
        return None

# Technische Indikatoren berechnen
def compute_indicators(df):
    df['SMA_10'] = df['price'].rolling(window=10).mean()
    df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()

    df['RSI'] = calculate_rsi(df)

    df['EMA_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Modelltraining
def make_prediction(df):
    if len(df) < 20:
        return None  # Nicht genug Daten
    features = df[['SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal']].dropna()
    targets = df['price'].shift(-1).dropna()
    X = features[:-1]
    y = targets.iloc[:len(X)]

    model = LinearRegression()
    model.fit(X, y)
    latest_features = features.iloc[-1:].values
    prediction = model.predict(latest_features)[0]
    return round(prediction, 2)

# Haupt-App
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")

    df = get_historical_btc_prices()
    if df is None:
        st.stop()

    df = compute_indicators(df)
    current_price = round(df['price'].iloc[-1], 2)
    prediction = make_prediction(df)

    st.markdown(f"**Aktueller Preis:** ${current_price}")

    if prediction is not None:
        st.markdown(f"**📉 Vorhersage für den nächsten Tag:** ${prediction}")
    else:
        st.warning("Nicht genügend Daten für Vorhersage.")

    st.subheader("🔍 Technische Indikatoren:")
    indicators = df.iloc[-1][['SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal']].round(2)
    for name, value in indicators.items():
        st.write(f"{name}: {value}")

    st.line_chart(df.set_index('timestamp')[['price', 'SMA_10', 'EMA_10']])

if __name__ == "__main__":
    app()
