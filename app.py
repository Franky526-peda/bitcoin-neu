import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seite konfigurieren
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# RSI-Berechnung ohne externe Bibliothek
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Aktuellen BTC-Preis von CoinDesk abrufen
def get_current_btc_price():
    url = "https://api.coindesk.com/v1/bpi/currentprice/USD.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = float(data['bpi']['USD']['rate'].replace(',', ''))
        return price
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Simulierte historische Preisdaten erzeugen (z. B. letzte 30 Minuten mit kleinen Schwankungen)
def generate_simulated_data(current_price, minutes=30):
    np.random.seed(42)
    prices = [current_price]
    for _ in range(minutes - 1):
        change = np.random.normal(0, 15)  # kleine Schwankung
        prices.append(max(0, prices[-1] + change))
    return pd.Series(prices[::-1])  # in umgekehrter Reihenfolge (älteste zuerst)

# Hauptfunktion der App
def main():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    current_price = get_current_btc_price()
    if current_price is None:
        st.stop()

    st.markdown("💰 **Aktueller Preis**")
    st.subheader(f"${current_price:,.2f}")

    # Historische Daten simulieren
    price_series = generate_simulated_data(current_price)
    rsi_series = calculate_rsi(price_series)

    # RSI anzeigen
    st.markdown("📊 **RSI der letzten 30 Minuten (simuliert)**")
    if rsi_series.isnull().all():
        st.info("RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")
    else:
        st.line_chart(rsi_series.dropna())
        latest_rsi = rsi_series.dropna().iloc[-1]
        st.metric("Letzter RSI-Wert", f"{latest_rsi:.2f}")

    # Vorhersage-Platzhalter (kann später durch ML-Modell ersetzt werden)
    st.markdown("📉 **Vorhersage**")
    st.info("Nicht genügend Daten für Vorhersage – Modell kann implementiert werden.")

if __name__ == "__main__":
    main()
