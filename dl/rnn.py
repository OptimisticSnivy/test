import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, DNN

# Load the dataset
df = pd.read_csv("Google_Stock_prices.csv")
prices = df["Open"].values
prices = prices.reshape(-1, 1)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)


# Create training sequences
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(i -window_size:i, 0)
        y.append(i, 0)
    return np.array(X), np.array(y)

window_size = 60
X , y = create_sequences(scaled_prices, window_size)

# Reshape X 
X = np.reshape(X.shape[0], (X.shape[1], 1))

# Build the model 
model = Sequential([
        SimpleRNN(units=50, activation='tanh', return_sequences=False,input_shape=(window_size,1)),
        Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=0.2)

# Predict
pred = model.predict(X)
pred = scaler.inverse_transform(pred)

# Plot
real = prices[window_size:]

# Plot real v pred
