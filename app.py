import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# Seite konfigurieren (muss ganz oben stehen!)
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Automatische Aktualisierung alle 60 Sekunden
st_autorefresh(interval=60 * 1000, key="refresh")

# Aktuellen Preis abrufen
def fetch_current_price():
    try:
        url = 'https://api.coincap.io/v2/assets/bitcoin'
        response = requests.get(url)
        data = response.json()
        return float(data['data']['priceUsd'])
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Historische Daten der letzten 30 Minuten abrufen (simuliert)
def fetch_historical_prices():
    try:
        url = 'https://api.coincap.io/v2/assets/bitcoin/history?interval=m1'
        params = {'start': int((datetime.utcnow().timestamp() - 1800) * 1000),
                  'end': int(datetime.utcnow().timestamp() * 1000)}
        response = requests.get(url, params=params)
        data = response.json()
        prices = [float(point['priceUsd']) for point in data['data']]
        timestamps = [datetime.fromtimestamp(point['time'] / 1000) for point in data['data']]
        return pd.DataFrame({'timestamp': timestamps, 'price': prices})
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der historischen Daten: {e}")
        return None

# RSI-Berechnung ohne externe Bibliothek
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Preisvorhersage (lineare Regression)
def predict_prices(prices, minutes_list=[1, 5, 10]):
    if len(prices) < 10:
        return None
    model = LinearRegression()
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values
    model.fit(X, y)
    future_predictions = {}
    for minute in minutes_list:
        future_index = len(prices) + minute
        predicted_price = model.predict([[future_index]])[0]
        future_predictions[minute] = predicted_price
    return future_predictions

# App Start
st.markdown("## 📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

current_price = fetch_current_price()
if current_price:
    st.subheader("💰 Aktueller Preis")
    st.markdown(f"<h1 style='color: green;'>${current_price:,.2f}</h1>", unsafe_allow_html=True)

    # Historische Daten abrufen
    df = fetch_historical_prices()
    if df is not None and not df.empty:
        st.subheader("📊 RSI der letzten 30 Minuten")
        df['price'] = pd.to_numeric(df['price'])
        df['rsi'] = calculate_rsi(df['price'])

        last_rsi = df['rsi'].iloc[-1] if not df['rsi'].isna().all() else None
        if last_rsi:
            st.write("Letzter RSI-Wert")
            st.metric("RSI", f"{last_rsi:.2f}")
        else:
            st.info("RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")

        # Preisvorhersage
        st.subheader("📉 Preisvorhersage")
        prediction = predict_prices(df['price'])
        if prediction:
            for minute, value in prediction.items():
                st.write(f"Vorhergesagter Preis in {minute} Minute(n): ${value:,.2f}")
        else:
            st.warning("Nicht genügend Daten für Vorhersage – Modell kann implementiert werden.")
    else:
        st.error("❌ Keine historischen Daten verfügbar.")
else:
    st.error("❌ Konnte aktuellen Preis nicht abrufen.")

