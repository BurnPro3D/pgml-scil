"""
Script: TensorFlow Installation Test with ConvLSTM Model

Description:
- This script verifies a successful TensorFlow installation by running a toy ConvLSTM model.
- It checks for GPU availability, generates random video-like data, and trains a small model.
- The model processes video sequences and performs classification on dummy data.
- It uses TensorFlow and Keras with ConvLSTM2D layers.

Author: Saqib Azim
Date: February 2025
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten
import numpy as np

# Suppress unnecessary TensorFlow logs (INFO and WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check TensorFlow version and GPU availability
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU Available:", gpus if gpus else "No GPU detected")

# Generate dummy video data (batch_size, timesteps, height, width, channels)
batch_size = 16
timesteps = 5
height = 64
width = 64
channels = 3
num_classes = 10

# Create random training and testing data
X_train = np.random.rand(500, timesteps, height, width, channels).astype(np.float32)
y_train = np.random.randint(0, num_classes, 500)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

X_test = np.random.rand(100, timesteps, height, width, channels).astype(np.float32)
y_test = np.random.randint(0, num_classes, 100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define a simple ConvLSTM model
model = Sequential([
    # ConvLSTM layer for spatio-temporal feature extraction
    ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
               return_sequences=True, input_shape=(timesteps, height, width, channels)),
    BatchNormalization(),

    # Second ConvLSTM layer with more filters
    ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
               return_sequences=False),
    BatchNormalization(),

    # Flatten and fully connected layers
    Flatten(),
    Dense(128, activation='relu'),         # Dense layer for feature processing
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model on dummy data
print("\nTraining model on dummy data...\n")
model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=1, 
          batch_size=batch_size)

# Print device placement details
print("\nDevice Placement Details:")
tf.debugging.set_log_device_placement(True)
