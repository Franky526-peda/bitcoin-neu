import streamlit as st
import pandas as pd
import requests
import time
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Muss GANZ OBEN stehen
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# Auto-refresh alle 60 Sekunden
st_autorefresh(interval=60 * 1000)

# CSV-Datei zur Speicherung
CSV_FILE = "btc_data.csv"

# RSI manuell berechnen
def compute_rsi(prices, window=14):
    if len(prices) < window:
        return None
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funktion: Bitcoin-Preis abrufen
def get_current_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Daten: {e}")
        return None

# Vorhersagefunktion mit einfachem ML-Modell
def make_prediction(df):
    if len(df) < 10:
        return None  # Nicht genug Daten
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Minutes'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 60
    model = LinearRegression()
    model.fit(df[['Minutes']], df['Price'])
    future_minutes = df['Minutes'].max() + 1
    return model.predict([[future_minutes]])[0]

# CSV lesen oder erstellen
def load_data():
    try:
        return pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Timestamp", "Price"])

# CSV speichern
def save_data(df):
    df.to_csv(CSV_FILE, index=False)

# Haupt-App
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit RSI")
    
    price = get_current_price()
    if price is None:
        st.warning("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df = load_data()
    new_row = pd.DataFrame([[timestamp, price]], columns=["Timestamp", "Price"])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)

    st.metric("💰 Aktueller Preis", f"${price:,.2f}")

    prediction = make_prediction(df)
    if prediction:
        st.metric("📈 Vorhersage in 1 Minute", f"${prediction:,.2f}")
    else:
        st.info("📉 Nicht genügend Daten für Vorhersage")

    # RSI berechnen
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    rsi_series = compute_rsi(df['Price'])
    latest_rsi = rsi_series.iloc[-1] if rsi_series is not None else None

    if latest_rsi:
        st.write(f"📊 Aktueller RSI: {latest_rsi:.2f}")
    else:
        st.write("📊 RSI wird berechnet… (mind. 14 Datenpunkte erforderlich)")

    # Verlauf anzeigen
    st.line_chart(df.set_index("Timestamp")["Price"])

if __name__ == "__main__":
    app()


