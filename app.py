import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import time

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st.title('💰 Bitcoin Predictor')
st.write("Diese App sagt den Bitcoin-Preis in 1, 5 und 10 Minuten voraus.")

# Funktion, um den Bitcoin-Preis zu holen
def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        st.write("API Antwort:", response.text)  # Debug-Ausgabe
        data = response.json()
        return data['bitcoin']['usd']
    except Exception as e:
        st.error(f"Fehler beim Abrufen des Bitcoin-Preises: {e}")
        return None

# Funktion für Vorhersage
def predict_price(data, minutes):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model.fit(X, y)
    prediction = model.predict(np.array([[len(data) + minutes]]))
    return prediction[0]

# Preise sammeln
prices = []
with st.spinner("Hole aktuelle Bitcoin-Preise (bitte Geduld, 1 Anfrage pro Minute)..."):
    for i in range(10):
        price = get_btc_price()
        if price is not None:
            prices.append(price)
