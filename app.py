import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Funktion, um Bitcoin-Preis von CoinGecko zu holen
def get_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data['bitcoin']['usd']

# Funktion für lineare Regression zur Vorhersage
def predict_price(data, minutes):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model.fit(X, y)
    prediction = model.predict(np.array([[len(data) + minutes]]))
    return prediction[0]

# Hauptprogramm
st.title('Bitcoin Predictor')

# Abrufen von Bitcoin-Preis für die letzten 10 Minuten
prices = []
for i in range(10):
    price = get_btc_price()
    prices.append(price)
    time.sleep(60)

# Vorhersage der nächsten Minuten
prediction_1_min = predict_price(prices, 1)
prediction_5_min = predict_price(prices, 5)
prediction_10_min = predict_price(prices, 10)

# Zeigen der aktuellen und prognostizierten Preise
st.write(f"Aktueller Bitcoin-Preis: ${prices[-1]:.2f}")
st.write(f"Vorhersage für den Bitcoin-Preis in 1 Minute: ${prediction_1_min:.2f}")
st.write(f"Vorhersage für den Bitcoin-Preis in 5 Minuten: ${prediction_5_min:.2f}")
st.write(f"Vorhersage für den Bitcoin-Preis in 10 Minuten: ${prediction_10_min:.2f}")
