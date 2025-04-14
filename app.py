import time
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Funktion zur Abfrage des aktuellen Bitcoin-Preises
def get_current_btc_price():
    url = "https://api.coindesk.com/v1/bpi/currentprice/USD.json"
    try:
        response = requests.get(url)
        data = response.json()
        current_price = data['bpi']['USD']['rate_float']
        return current_price
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen des aktuellen Preises: {e}")
        return None

# Funktion zur Abfrage historischer Bitcoin-Daten
def get_historical_btc_data():
    url = "https://api.coindesk.com/v1/bpi/historical/close.json"
    try:
        response = requests.get(url, params={'currency': 'USD', 'for_date': '2025-04-13'})
        data = response.json()
        return pd.DataFrame(data['bpi'], columns=['Date', 'Price'])
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen der historischen Daten: {e}")
        return None

# Funktion zur Berechnung des RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funktion zur Preisvorhersage
def make_prediction(df, minutes):
    # Feature: Zeit (minütlich), Preis
    df['Minute'] = np.arange(len(df))
    X = df[['Minute']]
    y = df['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_minute = np.array([[len(df) + minutes]])
    predicted_price = model.predict(future_minute)
    return predicted_price[0]

# Streamlit-App
def main():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # Abrufen des aktuellen Bitcoin-Preises
    current_price = get_current_btc_price()
    if current_price is None:
        st.stop()

    st.markdown("💰 **Aktueller Preis**")
    st.subheader(f"${current_price:,.2f}")

    # Abrufen der historischen Preisdaten
    historical_data = get_historical_btc_data()
    if historical_data is None:
        st.stop()

    # Berechnung des RSI
    rsi = calculate_rsi(historical_data['Price'])
    st.markdown("📊 **RSI der letzten 30 Minuten**")
    if rsi.isnull().all():
        st.info("RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")
    else:
        st.line_chart(rsi.dropna())
        latest_rsi = rsi.dropna().iloc[-1]
        st.metric("Letzter RSI-Wert", f"{latest_rsi:.2f}")

    # Vorhersage der nächsten Preisbewegung
    st.markdown("📉 **Preisvorhersage**")
    
    # Vorhersagen für 1, 5, und 10 Minuten
    pred_1 = make_prediction(historical_data, 1)
    pred_5 = make_prediction(historical_data, 5)
    pred_10 = make_prediction(historical_data, 10)

    st.subheader(f"Vorhergesagter Preis in 1 Minute: ${pred_1:,.2f}")
    st.subheader(f"Vorhergesagter Preis in 5 Minuten: ${pred_5:,.2f}")
    st.subheader(f"Vorhergesagter Preis in 10 Minuten: ${pred_10:,.2f}")

# Automatisches Refresh alle 60 Sekunden
def auto_refresh():
    st.experimental_rerun()

# Streamlit-Seite aktualisieren alle 60 Sekunden
if __name__ == "__main__":
    while True:
        main()
        time.sleep(60)  # 60 Sekunden warten, bevor die Daten erneut abgerufen werden
        auto_refresh()  # Seite automatisch neu laden
