import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# Muss an erster Stelle stehen!
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Seite wird alle 60 Sekunden automatisch neu geladen
st_autorefresh(interval=60 * 1000, key="datarefresh")

st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

# === Aktuellen Preis von CoinCap abrufen ===
def get_current_price():
    try:
        url = "https://api.coincap.io/v2/assets/bitcoin"
        response = requests.get(url)
        data = response.json()
        price = float(data['data']['priceUsd'])
        return round(price, 2)
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# === Historische Preisdaten der letzten 30 Minuten (1-Minuten-Auflösung) ===
def get_historical_data():
    try:
        url = "https://api.coincap.io/v2/assets/bitcoin/history"
        params = {
            "interval": "m1",
            "start": int((datetime.utcnow().timestamp() - 60 * 30) * 1000),
            "end": int(datetime.utcnow().timestamp() * 1000)
        }
        response = requests.get(url, params=params)
        data = response.json()
        prices = [float(item['priceUsd']) for item in data['data']]
        timestamps = [datetime.fromtimestamp(item['time'] / 1000) for item in data['data']]
        return pd.DataFrame({"timestamp": timestamps, "price": prices})
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der historischen Daten: {e}")
        return pd.DataFrame()

# === RSI-Berechnung ===
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === Preisvorhersage mit einfacher linearer Regression ===
def predict_prices(df):
    df = df.reset_index(drop=True)
    df['minute'] = np.arange(len(df))

    X = df[['minute']]
    y = df['price']

    model = LinearRegression().fit(X, y)

    preds = {
        1: round(model.predict([[len(df) + 1]])[0], 2),
        5: round(model.predict([[len(df) + 5]])[0], 2),
        10: round(model.predict([[len(df) + 10]])[0], 2)
    }
    return preds

# === Hauptlogik ===
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
