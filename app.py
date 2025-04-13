import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ---- Page config ----
st.set_page_config(page_title="Bitcoin Predictor", layout="centered")

# ---- Robust BTC data fetch ----
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "10",
        "interval": "hourly"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("⚠️ API-Rate-Limit erreicht. Bitte kurz warten und erneut versuchen.")
        elif response.status_code == 403:
            st.error("❌ Zugriff verweigert. Deine IP wurde eventuell blockiert.")
        else:
            st.error(f"❌ Fehler beim Abrufen der Daten: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Netzwerkfehler: {e}")
    return None

# ---- Technische Indikatoren ----
def calculate_indicators(df):
    df["SMA_10"] = df["price"].rolling(window=10).mean()
    df["EMA_10"] = df["price"].ewm(span=10, adjust=False).mean()

    # RSI ohne ta-Library
    delta = df["price"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    return df

# ---- Vorhersagefunktion ----
def make_prediction(df, minutes_ahead):
    if len(df) < 30:
        return "Nicht genügend Daten für Vorhersage"
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minutes"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 60

    X = df[["minutes"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    future_time = [[X["minutes"].max() + minutes_ahead]]
    prediction = model.predict(future_time)[0]
    return round(prediction, 2)

# ---- Streamlit App ----
def app():
    st.title("📈 Bitcoin Predictor – Live Vorhersagen mit erweiterten Features")

    data = get_btc_data()
    if not data or "prices" not in data:
        st.error("❌ Fehler beim Abrufen der Bitcoin-Daten. Bitte später erneut versuchen.")
        return

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = calculate_indicators(df)

    latest_price = df["price"].iloc[-1]
    pred_1 = make_prediction(df, 1)
    pred_5 = make_prediction(df, 5)
    pred_10 = make_prediction(df, 10)

    st.markdown(f"### 💰 Aktueller Preis: **${latest_price:,.2f}**")
    st.markdown("#### 📉 Preisvorhersagen")
    st.write(f"⏱️ In 1 Minute: **{pred_1}**")
    st.write(f"⏳ In 5 Minuten: **{pred_5}**")
    st.write(f"🕙 In 10 Minuten: **{pred_10}**")

    st.markdown("#### 📊 Technische Indikatoren (letzter Wert)")
    st.write(f"SMA (10): {df['SMA_10'].iloc[-1]:,.2f}")
    st.write(f"EMA (10): {df['EMA_10'].iloc[-1]:,.2f}")
    st.write(f"RSI: {df['RSI'].iloc[-1]:.2f}")
    st.write(f"MACD: {df['MACD'].iloc[-1]:.2f}")
    st.write(f"MACD Signal: {df['MACD_Signal'].iloc[-1]:.2f}")

    # Graphische Darstellung
    st.markdown("#### 📈 Preisverlauf")
    st.line_chart(df.set_index("timestamp")[["price", "SMA_10", "EMA_10"]])

# ---- Run ----
if __name__ == "__main__":
    app()

