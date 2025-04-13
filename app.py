import streamlit as st
import pandas as pd
import requests
import time
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from streamlit_autorefresh import st_autorefresh

# 🟢 Diese Zeile muss hier stehen (direkt nach den Imports!)
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# CSV-Datei zum Speichern der Daten
csv_file = "bitcoin_data.csv"

# Funktion zum Abrufen des aktuellen Bitcoin-Preises von CoinGecko
def get_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        print("Fehler beim Abrufen des Preises:", e)
        return None

# Funktion zur Preisvorhersage mit Linear Regression
def make_prediction(prices):
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    pred_1 = model.predict([[len(prices) + 1]])[0][0]
    pred_5 = model.predict([[len(prices) + 5]])[0][0]
    pred_10 = model.predict([[len(prices) + 10]])[0][0]

    return pred_1, pred_5, pred_10

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(price, pred_1, pred_5, pred_10):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Zeit": timestamp,
        "Preis": price,
        "Vorhersage_1min": pred_1,
        "Vorhersage_5min": pred_5,
        "Vorhersage_10min": pred_10
    }

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Hauptfunktion der Streamlit App
def app():
    # Auto-Refresh alle 60 Sekunden
    st_autorefresh(interval=60 * 1000, key="refresh")

    st.title("💹 Bitcoin Predictor – Live-Vorhersagen")
    st.markdown("Diese App sagt den Bitcoin-Preis für 1, 5 und 10 Minuten in die Zukunft voraus – basierend auf gesammelten Daten. Die Seite aktualisiert sich automatisch alle **60 Sekunden**.")

    price = get_btc_price()

    if price:
        # Daten einlesen oder initialisieren
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            prices = df["Preis"].tolist()
        else:
            df = pd.DataFrame()
            prices = []

        prices.append(price)

        # Vorhersage nur, wenn genug Daten vorhanden sind
        if len(prices) >= 2:
            pred_1, pred_5, pred_10 = make_prediction(prices)
        else:
            pred_1, pred_5, pred_10 = 0, 0, 0

        # Speichern
        save_to_csv(price, pred_1, pred_5, pred_10)

        # Aktualisierte Daten anzeigen
        df = pd.read_csv(csv_file)

        st.success(f"Aktueller Bitcoin-Preis: **${price:.2f}**")
        st.write(f"📈 Vorhersage in 1 Minute: **${pred_1:.2f}**")
        st.write(f"⏱️ Vorhersage in 5 Minuten: **${pred_5:.2f}**")
        st.write(f"⏳ Vorhersage in 10 Minuten: **${pred_10:.2f}**")

        st.markdown("---")
        st.markdown("### 📊 Verlauf der Preise und Vorhersagen")
        st.dataframe(df.tail(10))  # Nur letzte 10 Einträge anzeigen

        # Diagramm anzeigen
        st.markdown("### 📉 Verlauf als Diagramm")
        chart_data = df[["Zeit", "Preis", "Vorhersage_1min", "Vorhersage_5min", "Vorhersage_10min"]]
        chart_data["Zeit"] = pd.to_datetime(chart_data["Zeit"])
        chart_data = chart_data.set_index("Zeit")
        st.line_chart(chart_data)

    else:
        st.error("❌ Fehler beim Abrufen des Bitcoin-Preises.")

if __name__ == "__main__":
    app()
