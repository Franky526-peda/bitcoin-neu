import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Funktion zum Abrufen der historischen Bitcoin-Preisdaten von CoinGecko
def get_historical_btc_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '7',  # Abrufen der letzten 7 Tage (oder nach Bedarf anpassen)
    'interval': 'daily'  # Versuche 'daily' anstelle von 'minute'
}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'prices' in data:
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        st.error("Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return None

# Funktion zur Berechnung des RSI (Relative Strength Index) ohne 'ta' Modul
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Funktion zur Berechnung von SMA und EMA
def calculate_technical_indicators(df):
    # Berechnung des SMA (Simple Moving Average) mit einem Fenster von 10 Minuten
    df['SMA_10'] = df['price'].rolling(window=10).mean()
    
    # Berechnung des EMA (Exponential Moving Average) mit einem Fenster von 10 Minuten
    df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()

    # Berechnung des RSI
    df['RSI'] = calculate_rsi(df['price'])

    # Berechnung von MACD und Signal-Linie
    df['EMA_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Funktion zur Durchführung der Vorhersage mit LinearRegression
def make_prediction(df):
    if len(df) < 10:
        return None, None, None  # Nicht genügend Daten für Vorhersage

    # Feature und Ziel-Daten vorbereiten
    X = df[['price', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal']].dropna()
    y = df['price'].shift(-1).dropna()  # Vorhersage für den nächsten Preis

    model = LinearRegression()
    model.fit(X, y)
    
    pred_1 = model.predict(X.iloc[[-1]])[0]  # Vorhersage für 1 Minute
    return pred_1

# Funktion zum Speichern der Vorhersagehistorie
def save_to_csv(price, pred_1):
    data = {'price': [price], 'pred_1': [pred_1]}
    df = pd.DataFrame(data)
    df.to_csv('bitcoin_predictions.csv', mode='a', header=False, index=False)

# Streamlit App
def app():
    st.title("Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")

    # Aktuellen Bitcoin-Preis abrufen
    historical_data = get_historical_btc_prices()
    
    if historical_data is not None:
        # Berechnung der technischen Indikatoren
        historical_data = calculate_technical_indicators(historical_data)
        
        # Den letzten Preis abrufen
        current_price = historical_data['price'].iloc[-1]
        
        # Vorhersage berechnen
        pred_1 = make_prediction(historical_data)
        
        if pred_1 is None:
            st.write("Nicht genügend Daten für Vorhersage.")
        else:
            # Ergebnisse anzeigen
            st.write(f"Aktueller Preis: ${current_price:.2f}")
            st.write(f"Vorhersage für 1 Minute: ${pred_1:.2f}")

        # Historie der technischen Indikatoren und Vorhersagen anzeigen
        st.write("Technische Indikatoren:")
        st.write(f"SMA_10: {historical_data['SMA_10'].iloc[-1]:.2f}")
        st.write(f"EMA_10: {historical_data['EMA_10'].iloc[-1]:.2f}")
        st.write(f"RSI: {historical_data['RSI'].iloc[-1]:.2f}")
        st.write(f"MACD: {historical_data['MACD'].iloc[-1]:.2f}")
        st.write(f"MACD Signal: {historical_data['MACD_signal'].iloc[-1]:.2f}")
        
        # Ergebnisse speichern
        save_to_csv(current_price, pred_1)
        
        # Grafische Darstellung der Bitcoin-Preisdaten
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(historical_data['timestamp'], historical_data['price'], label='Bitcoin Preis')
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Preis (USD)')
        ax.set_title('Historische Bitcoin-Preise')
        st.pyplot(fig)

if __name__ == "__main__":
    app()
