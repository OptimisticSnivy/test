import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model = tf.keras.Sequential(
    [
        tf.kears.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.kears.layers.MaxPooling2D((2, 2)),
        tf.kears.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.kears.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

loss = model.evaluate(X_test, y_test)
