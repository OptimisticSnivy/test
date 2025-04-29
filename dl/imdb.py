# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
from tensoforflow.keras import layers

# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Pre-process the data
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=256)

# Build the model 
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=256),
    layers.Flatten(),
    layers.Dense(64, activation='relu')
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs =10, batch_size=512, validation_split=0.2, verbose=1 )

loss, accuracy = model.evaluate[X_test, y_test]

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
