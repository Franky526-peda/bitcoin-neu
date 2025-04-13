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

    # 1. Preis abrufen
    price = get_btc_price()

    if price:
        # 2. Lade bestehende Daten oder starte neu
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            prices = df['price'].tolist()
        else:
            df = pd.DataFrame()
            prices = []

        # 3. Neuen Preis zur Liste hinzufügen
        prices.append(price)

        # 4. Vorhersagen berechnen, nur wenn genug Daten vorhanden
        if len(prices) >= 2:
            pred_1, pred_5, pred_10 = make_prediction(prices)
        else:
            pred_1, pred_5, pred_10 = 0, 0, 0  # Platzhalter

        # 5. Speichern
        save_to_csv(price, pred_1, pred_5, pred_10)

        # 6. Daten neu laden (mit neuen Werten)
        df = pd.read_csv(csv_file)

        # 7. Ergebnisse anzeigen
        st.markdown(f"### 💰 Aktueller Bitcoin-Preis: **${price:.2f}**")
        st.markdown(f"- 📈 **Vorhersage in 1 Minute:** ${pred_1:.2f}")
        st.markdown(f"- ⏱️ **Vorhersage in 5 Minuten:** ${pred_5:.2f}")
        st.markdown(f"- ⏳ **Vorhersage in 10 Minuten:** ${pred_10:.2f}")

        # 8. Verlauf anzeigen
        st.write("### Verlauf:")
        st.dataframe(df)
    else:
        st.warning("Konnte aktuellen Bitcoin-Preis nicht abrufen.")
