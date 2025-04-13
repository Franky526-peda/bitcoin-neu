import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import os

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st.title('💰 Bitcoin Predictor')
st.write("Diese App sagt den Bitcoin-Preis in 1, 5 und 10 Minuten voraus und speichert die Ergebnisse.")

# 📂 Datei zum Speichern
csv_file = "bitcoin_prices.csv"

# 🟡 Hole aktuellen Preis
import requests

def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        response.raise_for_status()  # Überprüft, ob die Antwort erfolgreich war
        data = response.json()
        return data['bitcoin']['usd']
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen des Bitcoin-Preises von CoinGecko: {e}")
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

# 🔁 Preis sammeln und speichern
def save_to_csv(price, pred_1, pred_5, pred_10):
    new_data = {
        'timestamp': time.time(),
        'price': price,
        'pred_1': pred_1,
        'pred_5': pred_5,
        'pred_10': pred_10
    }
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df.append(new_data, ignore_index=True)
    else:
        df = pd.DataFrame([new_data])
    
    df.to_csv(csv_file, index=False)

# ⏳ Ablauf
if st.button('Start Vorhersage'):
    while True:
        price = get_btc_price()

        if price is not None:
            prices = simulate_historic_prices(price)
            
            pred_1 = predict_price(prices, 1)
            pred_5 = predict_price(prices, 5)
            pred_10 = predict_price(prices, 10)

            # Anzeigen
            st.success(f"Aktueller Bitcoin-Preis: ${price:.2f}")
            st.info(f"📈 Vorhersage in 1 Minute: ${pred_1:.2f}")
            st.info(f"📈 Vorhersage in 5 Minuten: ${pred_5:.2f}")
            st.info(f"📈 Vorhersage in 10 Minuten: ${pred_10:.2f}")

            # Speichern der Daten
            save_to_csv(price, pred_1, pred_5, pred_10)

            time.sleep(60)  # 1 Anfrage pro Minute (Rate Limit einhalten)
        else:
            st.error("Bitcoin-Preis konnte nicht abgerufen werden.")
            break
