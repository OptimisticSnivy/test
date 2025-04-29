import numpy as np
import pands as pd
import tensorflow as tf
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(128. activation='relu', input_shape=(X_train_shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

test_loss = model.evaluate(X_test, y_test)
print(f"MSE: {test_loss}")

y_pred = model.predict(X_test)

# Plot 
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("True")
plt.ylabel("Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
