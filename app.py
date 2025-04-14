import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Seite konfigurieren (muss ganz am Anfang stehen)
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Titel
st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")

# Funktion zur RSI-Berechnung
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funktion zum Abrufen historischer Bitcoin-Preise von CoinGecko
def get_btc_price_history():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1",
        "interval": "minutely"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        prices = data.get("prices", [])
        if not prices:
            st.error("⚠️ Fehler: Keine Preisdaten in API-Antwort enthalten.")
            return None

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Fehler beim Abrufen der Daten: {e}")
        return None

# Vorhersagefunktion mit einfachem linearen Modell
def predict_future_prices(df):
    df = df.copy()
    df["minute"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["minute"]], df["price"])

    pred_1 = model.predict([[len(df)]])[0]
    pred_5 = model.predict([[len(df) + 4]])[0]
    pred_10 = model.predict([[len(df) + 9]])[0]
    return pred_1, pred_5, pred_10

# Hauptfunktion
def app():
    df = get_btc_price_history()

    if df is None or df.empty:
        st.error("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return

    # Aktuellen Preis anzeigen
    current_price = df["price"].iloc[-1]
    st.subheader("💰 Aktueller Preis")
    st.metric(label="", value=f"${current_price:,.2f}")

    # RSI berechnen
    rsi_series = calculate_rsi(df["price"])
    latest_rsi = rsi_series.iloc[-1]

    if rsi_series.isna().sum() > 0:
        st.warning("📊 RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")
    else:
        st.subheader("📊 RSI")
        st.metric(label="Relative Strength Index (14)", value=f"{latest_rsi:.2f}")

    # Vorhersage anzeigen
    if len(df) >= 15:
        pred_1, pred_5, pred_10 = predict_future_prices(df)
        st.subheader("📉 Vorhersagen")
        st.write(f"⏱️ In 1 Minute: **${pred_1:,.2f}**")
        st.write(f"⏳ In 5 Minuten: **${pred_5:,.2f}**")
        st.write(f"🕙 In 10 Minuten: **${pred_10:,.2f}**")
    else:
        st.warning("📉 Nicht genügend Daten für Vorhersage")

    # Verlauf als Diagramm
    st.subheader("📈 Preisverlauf")
    st.line_chart(df["price"])

# App starten
if __name__ == "__main__":
    app()

