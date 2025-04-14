import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# Seite konfigurieren - muss ganz am Anfang stehen
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

def fetch_current_price():
    try:
        url = 'https://api.coincap.io/v2/assets/bitcoin'
        response = requests.get(url)
        data = response.json()

        if 'data' in data and 'priceUsd' in data['data']:
            return float(data['data']['priceUsd'])
        else:
            raise ValueError("API-Antwort enthält kein 'priceUsd'")
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

def fetch_historical_prices():
    try:
        url = 'https://api.coincap.io/v2/assets/bitcoin/history?interval=m1'
        params = {
            'start': int((datetime.utcnow().timestamp() - 1800) * 1000),
            'end': int(datetime.utcnow().timestamp() * 1000)
        }
        response = requests.get(url, params=params)
        data = response.json()

        if 'data' not in data or not data['data']:
            raise ValueError("Keine Daten in der API-Antwort gefunden")

        prices = [float(point['priceUsd']) for point in data['data']]
        timestamps = [datetime.fromtimestamp(point['time'] / 1000) for point in data['data']]
        return pd.DataFrame({'timestamp': timestamps, 'price': prices})

    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der historischen Daten: {e}")
        return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def make_prediction(df, minutes):
    df = df.copy()
    df['timestamp_ordinal'] = df['timestamp'].map(datetime.toordinal)
    X = df[['timestamp_ordinal']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    future_time = datetime.now() + timedelta(minutes=minutes)
    future_ordinal = np.array([[future_time.toordinal()]])
    predicted_price = model.predict(future_ordinal)[0]
    return predicted_price

def main():
    st_autorefresh(interval=60000, key="refresh")

    st.title("\U0001F4C8 Bitcoin Predictor – Live Vorhersagen mit RSI")

    current_price = fetch_current_price()
    if current_price:
        st.subheader("\U0001F4B0 Aktueller Preis")
        st.write(f"${current_price:,.2f}")
    else:
        st.warning("❌ Konnte aktuellen Preis nicht abrufen.")

    df = fetch_historical_prices()
    if df is not None and len(df) >= 15:
        st.subheader("\U0001F4CA RSI der letzten 30 Minuten")
        rsi_series = calculate_rsi(df['price'])
        last_rsi = rsi_series.iloc[-1]
        st.write(f"Letzter RSI-Wert: **{last_rsi:.2f}**")

        st.subheader("\U0001F4C9 Preisvorhersage")
        pred_1 = make_prediction(df, 1)
        pred_5 = make_prediction(df, 5)
        pred_10 = make_prediction(df, 10)

        st.write(f"Vorhergesagter Preis in 1 Minute: ${pred_1:,.2f}")
        st.write(f"Vorhergesagter Preis in 5 Minuten: ${pred_5:,.2f}")
        st.write(f"Vorhergesagter Preis in 10 Minuten: ${pred_10:,.2f}")
    else:
        st.warning("Nicht genügend Daten für RSI oder Vorhersage vorhanden.")

if __name__ == "__main__":
    main()
