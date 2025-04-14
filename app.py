import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# Seite konfigurieren
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st_autorefresh(interval=60 * 1000, key="datarefresh")

# 🟠 Funktion: Historische BTC-Preise (letzte 24h, minütlich)
def get_historical_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1",
        "interval": "minutely"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der historischen Daten: {e}")
        return None

# 🧮 RSI Berechnung (ohne externe Bibliotheken)
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# 🔮 Preisvorhersage mit Linear Regression
def predict_price(prices):
    if len(prices) < 10:
        return None
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future = np.array([[len(prices)]])
    prediction = model.predict(future)
    return prediction[0][0]

# 🔁 Hauptfunktion
def app():
    st.markdown("## 📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    df = get_historical_prices()

    if df is not None and not df.empty:
        current_price = df["price"].iloc[-1]
        st.markdown(f"### 💰 Aktueller Preis\n\n**${current_price:,.2f}**")

        # RSI berechnen
        rsi = calculate_rsi(df["price"])
        if rsi is not None:
            st.markdown(f"### 📊 RSI (14)\n\n**{rsi:.2f}**")
        else:
            st.markdown("📊 *RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)*")

        # Preisvorhersage
        prediction = predict_price(df["price"])
        if prediction is not None:
            st.markdown(f"### 🔮 Vorhersage (nächste Minute)\n\n**${prediction:,.2f}**")
        else:
            st.markdown("📉 *Nicht genügend Daten für Vorhersage*")
    else:
        st.error("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")

# ▶️ App starten
if __name__ == "__main__":
    app()

