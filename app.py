import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Ridge
from datetime import datetime, timedelta

# Automatisches Neuladen alle 60 Sekunden
st.query_params.update({"update": str(time.time())})
st.title("🔮 Live-Goldpreis-Vorhersage")

# Dummy-Goldpreis-Stream (ersetze später mit echter API)
@st.cache_data(ttl=60)
def get_live_gold_data():
    np.random.seed(int(time.time()) % 10000)
    now = datetime.now()
    prices = [2000 + np.sin(i/10) * 2 + np.random.normal(0, 0.5) for i in range(120)]
    timestamps = [now - timedelta(minutes=119-i) for i in range(120)]
    df = pd.DataFrame({"timestamp": timestamps, "price": prices})
    return df

df = get_live_gold_data()

# Visualisierung
st.line_chart(df.set_index("timestamp")["price"])

# Modelltraining
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

models = {}
predictions = {}

for minutes in [1, 5, 10]:
    model, df_features = train_model(df, minutes)
    last_input = df_features.iloc[-1][[f"lag_{i}" for i in range(1, 11)]].values.reshape(1, -1)
    pred = model.predict(last_input)[0]
    predictions[minutes] = pred

# Ausgabe der Prognosen
st.subheader("📈 Vorhersagen")
for minutes, value in predictions.items():
    st.write(f"**In {minutes} Minuten:** {value:.2f} USD")

st.caption("Daten werden alle 60 Sekunden automatisch aktualisiert.")

