import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# Automatische Aktualisierung alle 60 Sekunden
st_autorefresh(interval=60 * 1000, key="datarefresh")

# Streamlit-Seitenkonfiguration
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

st.title("\U0001F4C8 Bitcoin Predictor – Live Vorhersagen mit RSI")

# Funktion zum Abrufen des aktuellen Bitcoin-Preises von CoinCap
def fetch_current_price():
    url = "https://api.coincap.io/v2/assets/bitcoin"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['data']['priceUsd'])
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Funktion zum Simulieren historischer Daten (z. B. aus Cache oder Dummy-Generator)
def simulate_historical_prices(current_price):
    now = datetime.utcnow()
    timestamps = [now - pd.Timedelta(minutes=i) for i in range(30)][::-1]
    prices = [current_price + np.random.normal(0, 50) for _ in timestamps]
    df = pd.DataFrame({"timestamp": timestamps, "price": prices})
    return df

# RSI manuell berechnen (14 Perioden)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Preisvorhersage mit LinearRegression

def predict_future_prices(df):
    df = df.dropna().copy()
    if len(df) < 10:
        return None

    df['timestamp_ordinal'] = pd.to_datetime(df['timestamp']).map(datetime.toordinal)
    X = df['timestamp_ordinal'].values.reshape(-1, 1)
    y = df['price'].values

    model = LinearRegression()
    model.fit(X, y)

    future_minutes = [1, 5, 10]
    future_preds = {}
    now_ordinal = datetime.utcnow().toordinal()
    for minutes in future_minutes:
        future_time = now_ordinal + (minutes / (24 * 60))  # Umrechnen in Tagesbruchteil
        future_price = model.predict([[future_time]])[0]
        future_preds[minutes] = round(future_price, 2)
    return future_preds

# Hauptfunktion

def app():
    current_price = fetch_current_price()

    if current_price:
        st.subheader("\U0001F4B0 Aktueller Preis")
        st.markdown(f"<h2 style='color:green;'>${current_price:,.2f}</h2>", unsafe_allow_html=True)

        df = simulate_historical_prices(current_price)

        st.subheader("\U0001F4CA RSI der letzten 30 Minuten")
        df['rsi'] = calculate_rsi(df['price'])

        if df['rsi'].dropna().empty:
            st.warning("RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")
        else:
            last_rsi = df['rsi'].dropna().iloc[-1]
            st.markdown(f"**Letzter RSI-Wert**: `{last_rsi:.2f}`")

        st.subheader("\U0001F4C9 Preisvorhersage")
        predictions = predict_future_prices(df)

        if predictions:
            for minutes, price in predictions.items():
                st.write(f"Vorhergesagter Preis in {minutes} Minute(n): ${price:,.2f}")
        else:
            st.warning("Nicht genügend Daten für Vorhersage – Modell kann implementiert werden.")
    else:
        st.error("\u274C Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")

# App starten
if __name__ == "__main__":
    app()

