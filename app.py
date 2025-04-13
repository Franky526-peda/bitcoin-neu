import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# Seite konfigurieren – MUSS als erstes Streamlit-Kommando kommen!
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Automatisch jede Minute neu laden
st_autorefresh(interval=60 * 1000, key="auto_refresh")

# Funktion: RSI ohne externe Bibliothek
def compute_rsi(prices, window=14):
    deltas = prices.diff()
    gain = (deltas.where(deltas > 0, 0)).rolling(window=window).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Historische BTC-Daten abrufen
def get_historical_btc_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "10",
        "interval": "hourly"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    if "prices" not in data:
        return None
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# Technische Indikatoren berechnen
def add_indicators(df):
    df["SMA_10"] = df["price"].rolling(window=10).mean()
    df["EMA_10"] = df["price"].ewm(span=10, adjust=False).mean()

    df["EMA_12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI"] = compute_rsi(df["price"])
    return df

# Preisvorhersage mit einfacher Regression
def make_prediction(df, minutes_ahead):
    df = df.dropna()
    if len(df) < 20:
        return "Nicht genügend Daten für Vorhersage"
    df["timestamp_num"] = df.index.view(int)  # Zeit in ns
    X = df[["timestamp_num"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)

    future_time = df.index[-1] + pd.Timedelta(minutes=minutes_ahead)
    future_timestamp = int(future_time.value)
    prediction = model.predict([[future_timestamp]])
    return round(prediction[0], 2)

# App-Start
def app():
    st.title("📊 Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")

    df = get_historical_btc_prices()
    if df is None or df.empty:
        st.error("Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return

    df = add_indicators(df)
    latest_price = round(df["price"].iloc[-1], 2)
    st.markdown(f"**Aktueller Preis:** ${latest_price}")

    pred_1 = make_prediction(df, 1)
    pred_5 = make_prediction(df, 5)
    pred_10 = make_prediction(df, 10)

    st.markdown(f"📈 **Vorhersage für 1 Minute:** ${pred_1}")
    st.markdown(f"⏱️ **Vorhersage für 5 Minuten:** ${pred_5}")
    st.markdown(f"⏳ **Vorhersage für 10 Minuten:** ${pred_10}")

    st.subheader("Technische Indikatoren:")
    indicators = df.iloc[-1][["SMA_10", "EMA_10", "RSI", "MACD", "MACD_signal"]]
    for name, value in indicators.items():
        st.write(f"**{name}**: {round(value, 2) if not pd.isna(value) else 'Nicht verfügbar'}")

    st.subheader("📉 Preisverlauf")
    st.line_chart(df["price"])

if __name__ == "__main__":
    app()

