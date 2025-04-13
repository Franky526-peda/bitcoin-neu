import requests
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
import streamlit as st
from datetime import datetime
import os  # ❗️Dieser Import hat gefehlt


# 📂 Speicherort der CSV-Datei, um die Daten zu speichern
csv_file = "bitcoin_prices.csv"

# 📡 Abruf des Bitcoin-Preises von CoinGecko
def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['bitcoin']['usd']
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Abrufen des Bitcoin-Preises: {e}")
        return None

# 📝 Vorhersage basierend auf den gespeicherten Preisen
def make_prediction(prices):
    if len(prices) < 2:  # Um ein Modell zu trainieren, benötigen wir mindestens 2 Datenpunkte
        return 0, 0, 0  # Rückgabe von 0,0,0 als Platzhalter
    
    # X = Zeitstempel, Y = Preis
    X = [[i] for i in range(len(prices))]
    y = prices
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Vorhersagen für 1, 5 und 10 Minuten (1, 5, 10 Perioden nach dem letzten Punkt)
    pred_1 = model.predict([[len(prices) + 1]])[0]
    pred_5 = model.predict([[len(prices) + 5]])[0]
    pred_10 = model.predict([[len(prices) + 10]])[0]
    
    return pred_1, pred_5, pred_10

# 📝 Funktion zum Speichern der Preise und Vorhersagen in einer CSV-Datei
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
        new_df = pd.DataFrame([new_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])
    
    df.to_csv(csv_file, index=False)

# 🧑‍💻 Streamlit App
def app():
    st.title("Bitcoin Predictor")

    # Abrufen des aktuellen Bitcoin-Preises
    price = get_btc_price()

    if price:
        # Lade historische Daten (Preise)
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            prices = df['price'].tolist()
        else:
            prices = []

        # Füge den aktuellen Preis hinzu und mache eine Vorhersage
        prices.append(price)

        # Berechne Vorhersagen
        pred_1, pred_5, pred_10 = make_prediction(prices)

        # Speichere den aktuellen Preis und die Vorhersagen in der CSV
        save_to_csv(price, pred_1, pred_5, pred_10)

        # Zeige die Ergebnisse in Streamlit an
        st.write(f"**Aktueller Bitcoin-Preis**: ${price:.2f}")
        st.write(f"**Vorhersage für 1 Minute**: ${pred_1:.2f}")
        st.write(f"**Vorhersage für 5 Minuten**: ${pred_5:.2f}")
        st.write(f"**Vorhersage für 10 Minuten**: ${pred_10:.2f}")

        # Zeige die Historie der Preise und Vorhersagen
        st.write("### Historie der Preise und Vorhersagen:")
        st.write(df)

    # Automatisches Update jede Minute
    st.button("Aktualisieren", on_click=app)

if __name__ == "__main__":
    app()
