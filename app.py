import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st.title('💰 Bitcoin Predictor')
st.write("Diese App sagt den Bitcoin-Preis in 1, 5 und 10 Minuten voraus.")

# 🟡 Hole aktuellen Preis
def get_btc_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        st.error(f"Fehler beim Abrufen des Bitcoin-Preises von Binance: {e}")
        return None

# 🔵 Simuliere historische Preise (z. B. leicht schwankend um aktuellen Preis)
def simulate_historic_prices(current_price, num_points=10):
    np.random.seed(42)
    noise = np.random.normal(0, 10, size=num_points)
    return [current_price + n for n in noise]

# 🔮 Vorhersagefunktion
def predict_price(data, minutes):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model.fit(X, y)
    prediction = model.predict(np.array([[len(data) + minutes]]))
    return prediction[0]

# 🔁 Ablauf
price = get_btc_price()

if price is not None:
    prices = simulate_historic_prices(price)
    
    pred_1 = predict_price(prices, 1)
    pred_5 = predict_price(prices, 5)
    pred_10 = predict_price(prices, 10)

    st.success(f"Aktueller Bitcoin-Preis: ${price:.2f}")
    st.info(f"📈 Vorhersage in 1 Minute: ${pred_1:.2f}")
    st.info(f"📈 Vorhersage in 5 Minuten: ${pred_5:.2f}")
    st.info(f"📈 Vorhersage in 10 Minuten: ${pred_10:.2f}")

    st.line_chart(pd.DataFrame({
        "Historisch": prices + [np.nan]*10,
        "Vorhersage": [np.nan]*len(prices) + [pred_1, np.nan, np.nan, np.nan, pred_5, np.nan, np.nan, np.nan, np.nan, pred_10]
    }))
else:
    st.error("Bitcoin-Preis konnte nicht abgerufen werden.")
