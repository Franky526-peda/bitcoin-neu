import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Dein CoinMarketCap API-Schlüssel hier einfügen
API_KEY = 'DEIN_API_SCHLÜSSEL_HIER_EINFÜGEN'

# Header für alle Anfragen
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY,
}

# Funktion: aktuellen BTC-Preis abrufen
def get_current_price():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    params = {
        'symbol': 'BTC',
        'convert': 'USD'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        return data['data']['BTC']['quote']['USD']['price']
    except Exception as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Funktion: historische BTC-Daten abrufen (täglich)
def get_historical_data():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical'
    params = {
        'symbol': 'BTC',
        'convert': 'USD',
        'time_start': '2024-12-01',
        'time_end': datetime.now().strftime('%Y-%m-%d'),
        'interval': 'daily'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        quotes = data['data']['quotes']
        df = pd.DataFrame([{
            'timestamp': q['time_open'],
            'price': q['quote']['USD']['close']
        } for q in quotes])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return None

# Hauptfunktion der App
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # Aktueller Preis
    price = get_current_price()
    if price:
        st.subheader("💰 Aktueller Preis")
        st.metric(label="Bitcoin Preis (USD)", value=f"${price:,.2f}")

    # Historische Daten
    df = get_historical_data()
    if df is not None and not df.empty:
        st.subheader("📊 Preisentwicklung")
        st.line_chart(df['price'])
    else:
        st.warning("Keine historischen Daten verfügbar.")

# App starten
if __name__ == "__main__":
    app()
