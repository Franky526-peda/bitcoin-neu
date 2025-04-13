import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Abrufen von historischen Bitcoin-Preisen von CoinGecko
# Abrufen von historischen Bitcoin-Preisen von CoinGecko
def get_historical_btc_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "10",  # 10 Tage historische Daten
        "interval": "minute"  # Intervall in Minuten
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Ausgabe der vollständigen Antwort der API
        st.write("API-Antwort:", data)  # Zeigt die gesamte Antwort an
        
        # Prüfen, ob der "prices"-Schlüssel vorhanden ist
        if "prices" not in data:
            raise ValueError("Die API-Antwort enthält keinen 'prices'-Schlüssel.")
        
        prices = data["prices"]
        return [(datetime.utcfromtimestamp(timestamp / 1000), price) for timestamp, price in prices]
    
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen der Bitcoin-Daten: {e}")
        return []
    except ValueError as e:
        st.error(f"Fehler in den API-Daten: {e}")
        return []



# Berechnung der technischen Indikatoren
def calculate_indicators(df):
    df['SMA_10'] = df['Preis'].rolling(window=10).mean()  # 10-Minuten SMA
    df['EMA_10'] = df['Preis'].ewm(span=10, adjust=False).mean()  # 10-Minuten EMA

    delta = df['Preis'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA_12'] = df['Preis'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Preis'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# Vorhersage mit Linearer Regression unter Einbeziehung der Indikatoren
def make_prediction(df):
    # Entfernen von Zeilen mit NaN-Werten
    df = df.dropna()
    
    if len(df) < 2:  # Mindestens 2 Datenpunkte für die Vorhersage notwendig
        return "Nicht genügend Daten für Vorhersage"

    X = df[['Preis', 'SMA_10', 'EMA_10', 'RSI', 'MACD']]
    y = df['Preis']

    # Lineares Modell trainieren
    model = LinearRegression()
    model.fit(X, y)

    # Vorhersage für den nächsten Punkt
    future_data = pd.DataFrame({
        'Preis': [df['Preis'].iloc[-1]],  
        'SMA_10': [df['SMA_10'].iloc[-1]],
        'EMA_10': [df['EMA_10'].iloc[-1]],
        'RSI': [df['RSI'].iloc[-1]],
        'MACD': [df['MACD'].iloc[-1]],
    })

    prediction = model.predict(future_data)

    return prediction[0]

# Hauptfunktion
def app():
    st.title("Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")
    
    # Abruf von historischen Bitcoin-Daten
    historical_data = get_historical_btc_prices()

    if len(historical_data) == 0:
        st.error("Es konnten keine Bitcoin-Daten abgerufen werden. Bitte später erneut versuchen.")
        return

    # Erstellen des DataFrames mit historischen Preisen
    df = pd.DataFrame(historical_data, columns=["Zeit", "Preis"])

    # Berechnung der technischen Indikatoren
    df = calculate_indicators(df)

    # Berechnung der Vorhersage unter Einbeziehung der technischen Indikatoren
    pred_1 = make_prediction(df)

    # Anzeige der aktuellen Preisvorhersage und der technischen Indikatoren
    st.write(f"Aktueller Preis: {df['Preis'].iloc[-1]}")
    st.write(f"Vorhersage für 1 Minute: {pred_1}")

    st.write("Technische Indikatoren:")
    st.write(f"SMA_10: {df['SMA_10'].iloc[-1]}")
    st.write(f"EMA_10: {df['EMA_10'].iloc[-1]}")
    st.write(f"RSI: {df['RSI'].iloc[-1]}")
    st.write(f"MACD: {df['MACD'].iloc[-1]}")
    st.write(f"MACD Signal: {df['MACD_signal'].iloc[-1]}")

    # Anzeige eines Diagramms der Preisentwicklung
    st.line_chart(df.set_index('Zeit'))

if __name__ == "__main__":
    app()
