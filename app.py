import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSI
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

# Abrufen von historischen Bitcoin-Preisen von CoinGecko
def get_historical_btc_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "30",  # 30 Tage historische Daten
        "interval": "daily"  # Intervall in Tagen
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Prüfen, ob der "prices"-Schlüssel vorhanden ist
        if "prices" not in data:
            raise ValueError("Die API-Antwort enthält keinen 'prices'-Schlüssel.")
        
        prices = data["prices"]
        # Daten umwandeln in pandas DataFrame
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen der Bitcoin-Daten: {e}")
        return pd.DataFrame()  # Leerer DataFrame
    
    except ValueError as e:
        st.error(f"Fehler in den API-Daten: {e}")
        return pd.DataFrame()

# Berechnen der technischen Indikatoren
def calculate_indicators(df):
    # Berechnung des Simple Moving Average (SMA)
    sma = SMAIndicator(df['price'], window=10)
    df['SMA_10'] = sma.sma_indicator()

    # Berechnung des Exponential Moving Average (EMA)
    ema = EMAIndicator(df['price'], window=10)
    df['EMA_10'] = ema.ema_indicator()

    # Berechnung des Relative Strength Index (RSI)
    rsi = RSI(df['price'], window=14)
    df['RSI'] = rsi.rsi()

    # Berechnung des MACD
    macd = MACD(df['price'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    return df

# Vorhersagefunktion (Dummy-Funktion für die Vorhersage)
def make_prediction(df):
    if len(df) < 2:
        return "Nicht genügend Daten für Vorhersage"
    else:
        # Dummy-Vorhersage basierend auf letzten Preis
        latest_price = df['price'].iloc[-1]
        return latest_price * 1.01  # Beispielvorhersage, z.B. 1% Preisanstieg

# Streamlit App
def app():
    st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
    st.title("Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")
    
    # Abruf von historischen Bitcoin-Daten
    df = get_historical_btc_prices()
    
    if df.empty:
        return
    
    # Berechne technische Indikatoren
    df = calculate_indicators(df)
    
    # Anzeige des aktuellen Bitcoin-Preises
    current_price = df['price'].iloc[-1]
    st.write(f"Aktueller Preis: ${current_price:.2f}")
    
    # Vorhersage
    prediction = make_prediction(df)
    st.write(f"Vorhersage für 1 Minute: ${prediction:.2f}")
    
    # Anzeige der technischen Indikatoren
    st.write("Technische Indikatoren:")
    st.write(f"SMA_10: {df['SMA_10'].iloc[-1]:.2f}")
    st.write(f"EMA_10: {df['EMA_10'].iloc[-1]:.2f}")
    st.write(f"RSI: {df['RSI'].iloc[-1]:.2f}")
    st.write(f"MACD: {df['MACD'].iloc[-1]:.2f}")
    st.write(f"MACD Signal: {df['MACD_signal'].iloc[-1]:.2f}")
    
    # Grafische Darstellung
    st.subheader("Bitcoin Preis und technische Indikatoren")
    st.line_chart(df[['price', 'SMA_10', 'EMA_10']])
    
# Anwendung starten
if __name__ == "__main__":
    app()
