import streamlit as st
st.set_page_config(page_title="Live-Goldpreis", layout="centered")

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import requests
from streamlit_autorefresh import st_autorefresh

# 🔁 Automatisches Neuladen alle 60 Sekunden
st_autorefresh(interval=60 * 1000, key="auto-refresh")

# ✅ Dein MetalPriceAPI-Key
API_KEY = "8a263511db4d7c22d3e6c09a58397f16"

# 🔄 Live-Goldpreis abrufen
def get_live_gold_price():
    url = "https://api.metalpriceapi.com/v1/latest"
    params = {
        "api_key": API_KEY,
        "base": "USD",
        "currencies": "XAU"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Zeige API-Antwort zur Fehlersuche
    st.write("📡 API-Antwort:", data)
    
    if "rates" in data and "XAU" in data["rates"]:
        return 1 / data["rates"]["XAU"]  # Umrechnen zu USD/XAU
    else:
        return None

# 🚀 UI
st.title("📈 Live-Goldpreis mit Vorhersage (auto-refresh)")

# 📥 Live-Daten abrufen
price_now = get_live_gold_price()
timestamp = datetime.now()

if price_now is None:
    st.error("❌ Konnte aktuellen Goldpreis nicht abrufen.")
    st.stop()

# 📊 Simulierte Historie
np.random.seed(42)
noise = np.random.normal(0, 0.3, 120)
trend = np.linspace(0, 1.5, 120)
base = price_now - 1.5
prices = base + trend + noise
timestamps = [timestamp - pd.Timedelta(minutes=i) for i in range(119, -1, -1)]
df = pd.DataFrame({'timestamp': timestamps, 'price': prices})

# Aktuellen Preis anhängen
df = pd.concat([df, pd.DataFrame([{"timestamp": timestamp, "price": price_now}])], ignore_index=True)

# 🧠 Einfaches Modell
def train_model(data, horizon):
    df = data.copy()
    for i in range(1, 11):
        df[f"lag_{i}"] = df["price"].shift(i)
    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, 11)]]
    y = df["price"].shift(-horizon).dropna()
    X = X.iloc[:len(y)]
    model = Ridge()
    model.fit(X, y)
    return model, df

# 🔮 Vorhersagen
predictions = {}
for m in [1, 5, 10]:
    model, features = train_model(df, m)
    X_last = features.iloc[-1][[f"lag_{i}" for i in range(1, 11)]].values.reshape(1, -1)
    pred = model.predict(X_last)[0]
    predictions[m] = pred

# 🧾 Anzeige
st.metric("📍 Aktueller Goldpreis", f"{price_now:.2f} USD")

st.subheader("🔮 Vorhersagen")
for m, p in predictions.items():
    st.metric(f"In {m} Minuten", f"{p:.2f} USD")

st.subheader("📉 Verlauf (letzte 100 Minuten)")
st.line_chart(df.set_index("timestamp")["price"].tail(100))

st.caption("🔄 Diese App lädt sich automatisch alle 60 Sekunden neu.")

