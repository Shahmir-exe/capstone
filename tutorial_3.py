'''
Author: Shahmir Rizvi
Date: 10/8/2024
Description: Different ways to make a model. Tested with MNIST data set.
'''

import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 # The image size is 28x28
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

test_index = 0
test_image = x_test[test_index]
result = y_test[test_index]

def method_1():
    # Sequential API (Very convenient, not very flexible)
    model = keras.Sequential(
        [
            keras.Input(shape=(28*28)),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(10),
        ]
    )

    print(model.summary())

    # Specifies network configurations
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Specify concrete training
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

def method_2():
    # Sequential API (Very convenient, not very flexible)
    model = keras.Sequential()
    model.add(keras.Input(shape=(28*28)))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(10))

    # Specifies network configurations
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Specify concrete training
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

def method_3():
    # Functional API (A bit more flexible)
    inputs = keras.Input(shape=(28*28))
    x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
    x = layers.Dense(256, activation="relu", name="second_layer")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # Specifies network configurations
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Specify concrete training
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    print("Prediction: ", model.predict(np.expand_dims(x_test[0], axis=0)))
    print("Ground Truth: ", result)

def display_image():
    image = test_image * 255.0
    image = image.reshape(28,28)
    cv2.imshow('Digit', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


method_3()
display_image()