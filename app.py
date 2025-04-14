import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Seite konfigurieren
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Funktion: Preis von CoinGecko abrufen
def get_btc_price_public():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Daten: {e}")
        return None

# Funktion: RSI berechnen (ohne ta-lib oder ta)
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None

    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]

# Funktion: Dummy-Vorhersage
def predict_price(prices):
    if len(prices) < 5:
        return None  # Nicht genug Daten
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future = np.array([[len(prices)]])  # nächster Zeitschritt
    prediction = model.predict(future)
    return prediction[0][0]

# Hauptfunktion
def app():
    st.markdown("## 📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

    # SessionState verwenden für Preis-Log
    if "prices" not in st.session_state:
        st.session_state.prices = []

    current_price = get_btc_price_public()
    if current_price is not None:
        st.session_state.prices.append(current_price)

        st.markdown(f"### 💰 Aktueller Preis\n\n**${current_price:,.2f}**")

        # Vorhersage
        prices_series = pd.Series(st.session_state.prices)
        prediction = predict_price(prices_series)

        if prediction is not None:
            st.markdown(f"### 🔮 Vorhersage (nächste Minute)\n\n**${prediction:,.2f}**")
        else:
            st.markdown("📉 *Nicht genügend Daten für Vorhersage*")

        # RSI-Berechnung
        rsi = calculate_rsi(prices_series)
        if rsi is not None:
            st.markdown(f"### 📊 RSI (14)\n\n**{rsi:.2f}**")
        else:
            st.markdown("📊 *RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)*")

    else:
        st.error("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")

# Streamlit ausführen
if __name__ == "__main__":
    app()
