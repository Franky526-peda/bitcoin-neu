import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Simulation von Goldpreisdaten (1.000 Minuten)
np.random.seed(42)
n = 1000
base_price = 2000
trend = np.linspace(0, 10, n)
noise = np.random.normal(0, 1, n)
prices = base_price + trend + noise
df = pd.DataFrame({'price': prices})

# Feature-Engineering: Letzte 5 Minuten als Input
window = 5
for i in range(1, window + 1):
    df[f'lag_{i}'] = df['price'].shift(i)

df = df.dropna()

# Trainingsdaten
X = df[[f'lag_{i}' for i in range(1, window + 1)]]
y = df['price']

model = RandomForestRegressor()
model.fit(X, y)

# Vorhersage der nächsten Minute
last_values = df.iloc[-1][[f'lag_{i}' for i in range(1, window + 1)]].values.reshape(1, -1)
prediction = model.predict(last_values)[0]

# Streamlit UI
st.title("Goldpreis-Vorhersage (1 Minute in die Zukunft)")
st.line_chart(df['price'][-100:])
st.write(f"**Prognostizierter Goldpreis in 1 Minute:** `{prediction:.2f} USD`")

