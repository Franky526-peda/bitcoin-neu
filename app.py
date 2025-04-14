import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Twelve Data API-Key
API_KEY = "1c83ee150f8344eaa397d1d90a9da4f4"

# Page config
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Funktion zur Preisabfrage
def get_current_price():
    url = f"https://api.twelvedata.com/price?symbol=BTC/USD&apikey={API_KEY}"
    response = requests.get(url).json()
    try:
        return float(response["price"])
    except KeyError:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {response}")
        return None

# Funktion für historische Minutenpreise
def get_historical_prices(minutes=30):
    interval = "1min"
    outputsize = str(minutes)
    url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    response = requests.get(url).json()
    try:
        df = pd.DataFrame(response["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)
        df = df.sort_values("datetime")
        return df
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {response}")
        return None

# RSI-Berechnung
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Preisvorhersage mit linearer Regression
def predict_prices(data, forecast_minutes=[1, 5, 10]):
    df = data.copy()
    df["minutes"] = np.arange(len(df))
    X = df[["minutes"]]
    y = df["close"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = {}
    last_minute = df["minutes"].iloc[-1]
    for minutes in forecast_minutes:
        future_minute = [[last_minute + minutes]]
        predicted_price = model.predict(future_minute)[0]
        predictions[minutes] = predicted_price

    return predictions

# Hauptfunktion
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    price = get_current_price()
    if price:
        st.subheader("💰 Aktueller Preis:")
        st.metric("Bitcoin Preis (USD)", f"${price:,.2f}")

    historical_df = get_historical_prices(30)
    if historical_df is not None and not historical_df.empty:
        st.subheader("📊 RSI der letzten 30 Minuten")
        rsi_series = calculate_rsi(historical_df["close"])
        if not rsi_series.empty:
            st.write(f"Letzter RSI-Wert: {rsi_series.iloc[-1]:.2f}")
        else:
            st.warning("Nicht genügend Daten für RSI-Berechnung.")

        st.subheader("📉 Preisvorhersage mit linearer Regression")
        predictions = predict_prices(historical_df)
        for minutes, forecast in predictions.items():
            st.write(f"Vorhergesagter Preis in {minutes} Minute(n): ${forecast:,.2f}")
    else:
        st.warning("Keine historischen Preisdaten verfügbar – keine Vorhersage möglich.")

    st.write("🔄 Aktualisiert sich alle 60 Sekunden ...")

# Automatische Aktualisierung
def auto_refresh():
    time.sleep(60)
    st.experimental_rerun()

# Starte App
if __name__ == "__main__":
    app()
    auto_refresh()
