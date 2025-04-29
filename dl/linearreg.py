# Step 1: Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Step 2: Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Step 3: Preprocess the Data
# Standardize the features (mean = 0, std = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Build the Deep Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Single neuron output for regression
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Step 7: Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Visualize the results
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('True House Prices')
plt.ylabel('Predicted House Prices')
plt.title('California Housing Price Prediction')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()
