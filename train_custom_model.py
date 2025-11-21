"""
Custom Handwritten Digit Recognition - Model Training Script
Author: Htoo Aunt
Description: Train a CNN model on custom handwriting data with optional MNIST augmentation
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import glob

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("CUSTOM HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'custom_data_dir': 'custom_data',
    'include_mnist': True,           # Set to False to train ONLY on custom data
    'mnist_samples_per_digit': 1000, # How many MNIST samples to include per digit (0 = all)
    'use_transfer_learning': False,  # Set to True to fine-tune existing model
    'existing_model_path': 'model/digit_model.h5',
    'output_model_name': 'model/digit_model.h5',  # Changed to match Flask app
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
}

print("\nConfiguration:")
print(f"  - Include MNIST data: {CONFIG['include_mnist']}")
print(f"  - Use transfer learning: {CONFIG['use_transfer_learning']}")
print(f"  - Epochs: {CONFIG['epochs']}")

# ============================================
# STEP 1: LOAD CUSTOM DATA
# ============================================
print("\n[1] Loading Custom Dataset...")

def load_custom_images(data_dir, split='train'):
    """Load custom images from directory structure."""
    X = []
    y = []

    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        print(f"  Warning: {split_dir} not found")
        return np.array([]), np.array([])

    for digit in range(10):
        digit_dir = os.path.join(split_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue

        # Load all image formats
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(digit_dir, pattern)))

        for img_path in image_files:
            try:
                # Load and preprocess image
                img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
                img_array = img_to_array(img)

                # Invert if needed (make sure digit is white on black for model)
                # Check if image has light background (digit on white)
                if np.mean(img_array) > 127:
                    img_array = 255 - img_array

                X.append(img_array)
                y.append(digit)
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")

    return np.array(X), np.array(y)

# Load custom training and test data
X_custom_train, y_custom_train = load_custom_images(CONFIG['custom_data_dir'], 'train')
X_custom_test, y_custom_test = load_custom_images(CONFIG['custom_data_dir'], 'test')

print(f"  Custom training samples: {len(X_custom_train)}")
print(f"  Custom test samples: {len(X_custom_test)}")

# Check if we have enough custom data
if len(X_custom_train) == 0:
    print("\n" + "!" * 60)
    print("ERROR: No custom training data found!")
    print("!" * 60)
    print("\nPlease add images to the custom_data/train/ folders.")
    print("Run 'python setup_custom_data.py' to create the folder structure.")
    print("!" * 60)
    exit(1)

# Count samples per digit
print("\n  Samples per digit (training):")
for digit in range(10):
    count = np.sum(y_custom_train == digit)
    status = "OK" if count >= 10 else "LOW" if count > 0 else "NONE"
    print(f"    Digit {digit}: {count} samples [{status}]")

# ============================================
# STEP 2: OPTIONALLY LOAD MNIST DATA
# ============================================
if CONFIG['include_mnist']:
    print("\n[2] Loading MNIST Dataset to augment training...")

    (X_mnist_train, y_mnist_train), (X_mnist_test, y_mnist_test) = mnist.load_data()

    # Reshape MNIST to match our format
    X_mnist_train = X_mnist_train.reshape(-1, 28, 28, 1)
    X_mnist_test = X_mnist_test.reshape(-1, 28, 28, 1)

    # Sample MNIST if configured
    if CONFIG['mnist_samples_per_digit'] > 0:
        X_mnist_sampled = []
        y_mnist_sampled = []

        for digit in range(10):
            digit_indices = np.where(y_mnist_train == digit)[0]
            n_samples = min(CONFIG['mnist_samples_per_digit'], len(digit_indices))
            selected = np.random.choice(digit_indices, n_samples, replace=False)

            X_mnist_sampled.append(X_mnist_train[selected])
            y_mnist_sampled.append(y_mnist_train[selected])

        X_mnist_train = np.concatenate(X_mnist_sampled)
        y_mnist_train = np.concatenate(y_mnist_sampled)

    print(f"  MNIST training samples: {len(X_mnist_train)}")
    print(f"  MNIST test samples: {len(X_mnist_test)}")

    # Combine custom and MNIST data
    X_train = np.concatenate([X_custom_train, X_mnist_train])
    y_train = np.concatenate([y_custom_train, y_mnist_train])

    if len(X_custom_test) > 0:
        X_test = np.concatenate([X_custom_test, X_mnist_test])
        y_test = np.concatenate([y_custom_test, y_mnist_test])
    else:
        X_test = X_mnist_test
        y_test = y_mnist_test
        print("  Using MNIST test set (no custom test data)")
else:
    print("\n[2] Skipping MNIST (training only on custom data)...")
    X_train = X_custom_train
    y_train = y_custom_train

    if len(X_custom_test) > 0:
        X_test = X_custom_test
        y_test = y_custom_test
    else:
        # Split training data if no test data provided
        print("  No test data - splitting training data 80/20")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

print(f"\n  Total training samples: {len(X_train)}")
print(f"  Total test samples: {len(X_test)}")

# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================
print("\n[3] Preprocessing Data...")

# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Ensure correct shape
if len(X_train.shape) == 3:
    X_train = X_train.reshape(-1, 28, 28, 1)
if len(X_test.shape) == 3:
    X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

# Shuffle training data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train_encoded = y_train_encoded[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"  Training data shape: {X_train.shape}")
print(f"  Test data shape: {X_test.shape}")
print(f"  Data normalized to range [0, 1]")

# ============================================
# STEP 4: BUILD OR LOAD MODEL
# ============================================
if CONFIG['use_transfer_learning'] and os.path.exists(CONFIG['existing_model_path']):
    print("\n[4] Loading Existing Model for Transfer Learning...")

    model = load_model(CONFIG['existing_model_path'])

    # Freeze early layers (optional - uncomment to freeze)
    # for layer in model.layers[:-2]:
    #     layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"  Loaded model from: {CONFIG['existing_model_path']}")
    print(f"  Using learning rate: {CONFIG['learning_rate'] / 10}")

else:
    print("\n[4] Building New CNN Model...")

    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),

        # Flatten and Dense Layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),

        # Output Layer
        layers.Dense(10, activation='softmax', name='output')
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Display model architecture
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
model.summary()

# ============================================
# STEP 5: TRAIN THE MODEL WITH DATA AUGMENTATION
# ============================================
print("\n[5] Training Model with Data Augmentation...")

# Create data augmentation layer to handle different drawing styles
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),  # ±10% rotation
    layers.RandomZoom(0.1),      # ±10% zoom
    layers.RandomTranslation(0.1, 0.1),  # ±10% shift
])

# Build augmented model for training
augmented_model = keras.Sequential([
    data_augmentation,
    model
])

# Compile the augmented model
augmented_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = augmented_model.fit(
    X_train, y_train_encoded,
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['epochs'],
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("\n  Training completed!")

# ============================================
# STEP 6: EVALUATE THE MODEL
# ============================================
print("\n[6] Evaluating Model on Test Set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

print(f"\n{'=' * 60}")
print("FINAL RESULTS")
print(f"{'=' * 60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred_classes,
                          target_names=[str(i) for i in range(10)]))

# ============================================
# STEP 7: SAVE THE MODEL
# ============================================
print("\n[7] Saving Model...")

# Save only to digit_model.h5 (single model for Flask app)
model.save('model/digit_model.h5')
print(f"  Model saved as 'model/digit_model.h5'")

# ============================================
# STEP 8: VISUALIZE RESULTS
# ============================================
print("\n[8] Creating Visualizations...")

# Plot 1: Training History
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig('model/custom_training_results.png', dpi=300, bbox_inches='tight')
print("  Visualizations saved as 'model/custom_training_results.png'")

# ============================================
# STEP 9: SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"  Model saved: {CONFIG['output_model_name']}")
print(f"  Training results: model/custom_training_results.png")
print(f"  Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nYou can now use this model in your Flask backend!")
print("=" * 60)

plt.show()
