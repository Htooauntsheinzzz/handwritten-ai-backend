"""
Simple MNIST Model Training Script
Author: Htoo Aunt
Description: Quick script to train a digit recognition model on MNIST only
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import os

# Create model directory
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("TRAINING DIGIT RECOGNITION MODEL (MNIST)")
print("=" * 60)

# Load MNIST data
print("\n[1] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# Build model
print("\n[2] Building CNN model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("\n[3] Training model...")
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate
print("\n[4] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'=' * 60}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"{'=' * 60}")

# Save
print("\n[5] Saving model...")
model.save('model/digit_model.h5')
print("  Model saved as 'model/digit_model.h5'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nYou can now start the Flask server:")
print("  python app.py")
print("=" * 60)
