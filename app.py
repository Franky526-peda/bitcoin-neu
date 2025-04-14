import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulierte Goldpreise (z. B. 1000 Minuten)
np.random.seed(42)
n_minutes = 1000
base_price = 2000
noise = np.random.normal(0, 1, n_minutes)
trend = np.linspace(0, 10, n_minutes)
gold_prices = base_price + trend + noise

# DataFrame erstellen
df = pd.DataFrame(gold_prices, columns=["price"])

# Skalierung
scaler = MinMaxScaler()
df["scaled_price"] = scaler.fit_transform(df[["price"]])

# Sequenzen erstellen
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(df["scaled_price"].values, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM-Modell
model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training
model.fit(X, y, epochs=10, batch_size=32)

# Letzte Sequenz vorhersagen
last_sequence = df["scaled_price"].values[-window_size:]
last_sequence = last_sequence.reshape((1, window_size, 1))
predicted_scaled = model.predict(last_sequence)[0][0]
predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

print(f"Prognostizierter Goldpreis in 1 Minute: {predicted_price:.2f} USD")

