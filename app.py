import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Seite konfigurieren
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# RSI ohne externe Bibliotheken berechnen
def calculate_rsi(prices: pd.Series, period: int = 14):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain).rolling(window=period).mean()
    loss = pd.Series(loss).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Historische Daten abrufen (letzte 2 Tage)
def get_historical_prices():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "2", "interval": "minute"}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            st.error(f"Fehler beim Abrufen der Daten: {response.status_code}")
            return None

        data = response.json()
        prices = data.get("prices", [])
        if not prices:
            st.error("Die API-Antwort enthält keine 'prices'-Daten.")
            return None

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        st.error(f"Unerwarteter Fehler beim Abrufen der Daten: {e}")
        return None

# Preisvorhersage (einfaches lineares Modell)
def predict_future_prices(df):
    if len(df) < 15:
        return None

    df = df.copy()
    df["minute"] = np.arange(len(df))
    model = LinearRegression()
    X = df[["minute"]]
    y = df["price"]
    model.fit(X, y)

    future_minutes = np.array([[len(df) + i] for i in [1, 5, 10]])
    predictions = model.predict(future_minutes)
    return predictions

# App starten
def main():
    st.markdown("## 📈 Bitcoin Predictor – Live Vorhersagen mit RSI")
    df = get_historical_prices()

    if df is None:
        return

    latest_price = df["price"].iloc[-1]
    st.markdown(f"### 💰 Aktueller Preis\n\n${latest_price:,.2f}")

    # RSI berechnen
    rsi = calculate_rsi(df["price"])
    if rsi.isna().all():
        st.warning("\n\n📊 RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")
    else:
        st.markdown(f"\n\n**RSI (14)**: {rsi.iloc[-1]:.2f}")

    # Vorhersage
    predictions = predict_future_prices(df)
    if predictions is None:
        st.warning("\n\n📉 Nicht genügend Daten für Vorhersage")
    else:
        st.markdown(f"\n\n**Vorhersage:**")
        st.write(f"In 1 Minute: ${predictions[0]:,.2f}")
        st.write(f"In 5 Minuten: ${predictions[1]:,.2f}")
        st.write(f"In 10 Minuten: ${predictions[2]:,.2f}")

    # Verlauf anzeigen
    with st.expander("Preisdaten anzeigen"):
        st.line_chart(df["price"])

if __name__ == "__main__":
    main()
