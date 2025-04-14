import streamlit as st
import requests
import numpy as np
import random
import time

# Dein API-Key
API_KEY = 'f8d360d4-146c-4f00-8a05-b2f156224c2a'
BASE_URL = "https://pro-api.coinmarketcap.com/v1"

# Aktuellen BTC-Preis abrufen
def get_current_price():
    url = f"{BASE_URL}/cryptocurrency/listings/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': API_KEY
    }
    params = {
        'start': '1',
        'limit': '1',
        'convert': 'USD'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        return data['data'][0]['quote']['USD']['price']
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Historische Minutenpreise der letzten 30 Minuten abrufen
def get_historical_data():
    url = f"{BASE_URL}/cryptocurrency/ohlcv/historical"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': API_KEY
    }
    params = {
        'symbol': 'BTC',
        'convert': 'USD',
        'interval': '1m',
        'time_end': int(time.time()),
        'time_start': int(time.time()) - 1800  # 30 Minuten zurück
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if "data" in data and "quotes" in data["data"]:
            return data["data"]["quotes"]
        else:
            st.warning("Keine historischen Daten verfügbar.")
            return []
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return []

# RSI berechnen
def calculate_rsi(prices, window=14):
    if len(prices) < window + 1:
        return None
    delta = np.diff(prices)
    gain = np.mean([d for d in delta if d > 0])
    loss = np.mean([-d for d in delta if d < 0])
    if loss == 0:
        return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Beispielhafte Vorhersage (Zufall)
def predict_price():
    return random.uniform(80000, 90000)

# Streamlit App
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    price = get_current_price()
    if price:
        st.subheader("💰 Aktueller Preis:")
        st.markdown(f"<h2 style='color: green;'>${price:,.2f}</h2>", unsafe_allow_html=True)

    data = get_historical_data()
    if data:
        prices = [entry['quote']['USD']['close'] for entry in data]
        rsi = calculate_rsi(prices)
        if rsi:
            st.subheader("📊 RSI der letzten 30 Minuten:")
            st.write(f"Letzter RSI-Wert: {rsi:.2f}")
        else:
            st.warning("Nicht genug Daten zur Berechnung des RSI.")

        # Vorhersagen
        st.subheader("🔮 Preisvorhersagen:")
        st.write(f"1 Minute: ${predict_price():,.2f}")
        st.write(f"5 Minuten: ${predict_price():,.2f}")
        st.write(f"10 Minuten: ${predict_price():,.2f}")

    else:
        st.warning("Keine historischen Preisdaten verfügbar – keine Vorhersage möglich.")

    # Automatische Aktualisierung (manuell besser als sofort rerun)
    st.markdown("🔄 Aktualisiert sich alle 60 Sekunden ...")
    time.sleep(60)
    st.experimental_rerun()

if __name__ == "__main__":
    app()

