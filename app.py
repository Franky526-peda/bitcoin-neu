import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st_autorefresh(interval=60 * 1000, key="refresh")  # 1x pro Minute aktualisieren

st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

# === Aktuellen Preis von CoinGecko ===
def get_current_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# === Historische Daten von CoinGecko ===
def get_historical_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "1", "interval": "minutely"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.tail(30)  # letzte 30 Minuten
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der historischen Daten: {e}")
        return pd.DataFrame()

# === RSI berechnen ===
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === Preisvorhersage mit Linear Regression ===
def predict_prices(df):
    df = df.reset_index(drop=True)
    df["minute"] = np.arange(len(df))
    X = df[["minute"]]
    y = df["price"]
    model = LinearRegression().fit(X, y)

    return {
        1: round(model.predict([[len(df) + 1]])[0], 2),
        5: round(model.predict([[len(df) + 5]])[0], 2),
        10: round(model.predict([[len(df) + 10]])[0], 2)
    }

# === Hauptfunktion ===
def main():
    price = get_current_price()
    if price:
        st.subheader("💰 Aktueller Preis")
        st.write(f"${price:,.2f}")
    else:
        st.warning("❌ Konnte aktuellen Preis nicht abrufen.")

    df = get_historical_data()
    if not df.empty and len(df) >= 15:
        st.subheader("📊 RSI der letzten 30 Minuten")
        rsi_series = calculate_rsi(df["price"])
        last_rsi = round(rsi_series.iloc[-1], 2)
        st.metric(label="Letzter RSI-Wert", value=last_rsi)

        st.subheader("📉 Preisvorhersage")
        predictions = predict_prices(df)
        st.write(f"**Vorhergesagter Preis in 1 Minute:** ${predictions[1]:,.2f}")
        st.write(f"**Vorhergesagter Preis in 5 Minuten:** ${predictions[5]:,.2f}")
        st.write(f"**Vorhergesagter Preis in 10 Minuten:** ${predictions[10]:,.2f}")
    else:
        st.warning("Nicht genügend Daten für RSI oder Vorhersage vorhanden.")

if __name__ == "__main__":
    main()

