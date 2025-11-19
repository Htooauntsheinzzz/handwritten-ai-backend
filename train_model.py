"""
Handwritten Digit Recognition - Model Training Script
Author: Htoo Aunt
Description: Train a CNN model on MNIST dataset for digit recognition
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
print("=" * 60)

# ============================================
# STEP 1: LOAD MNIST AND CUSTOM DATASETS
# ============================================
print("\n[1] Loading MNIST Dataset...")

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✓ MNIST Training samples: {X_train.shape[0]}")
print(f"✓ MNIST Testing samples: {X_test.shape[0]}")
print(f"✓ Image shape: {X_train.shape[1]}x{X_train.shape[2]}")

# Load custom data if available
import glob
from keras.preprocessing.image import load_img, img_to_array

def load_custom_images(data_dir, split='train'):
    """Load custom images from directory structure."""
    X = []
    y = []
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        return np.array([]), np.array([])

    for digit in range(10):
        digit_dir = os.path.join(split_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue

        patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(digit_dir, pattern)))

        for img_path in image_files:
            try:
                img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
                img_array = img_to_array(img).squeeze()
                if np.mean(img_array) > 127:
                    img_array = 255 - img_array
                X.append(img_array)
                y.append(digit)
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")

    return np.array(X), np.array(y)

# Check for custom data
custom_data_dir = 'custom_data'
X_custom_train, y_custom_train = load_custom_images(custom_data_dir, 'train')
X_custom_test, y_custom_test = load_custom_images(custom_data_dir, 'test')

if len(X_custom_train) > 0:
    print(f"✓ Custom Training samples: {len(X_custom_train)}")
    print(f"✓ Custom Testing samples: {len(X_custom_test)}")

    # Combine MNIST and custom data
    X_train = np.concatenate([X_train, X_custom_train])
    y_train = np.concatenate([y_train, y_custom_train])

    if len(X_custom_test) > 0:
        X_test = np.concatenate([X_test, X_custom_test])
        y_test = np.concatenate([y_test, y_custom_test])

    print(f"✓ Combined Training samples: {len(X_train)}")
    print(f"✓ Combined Testing samples: {len(X_test)}")
else:
    print("  (No custom data found - using MNIST only)")

print(f"✓ Number of classes: {len(np.unique(y_train))}")

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n[2] Preprocessing Data...")

# Reshape data to add channel dimension (for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

print(f"✓ Training data shape: {X_train.shape}")
print(f"✓ Training labels shape: {y_train_encoded.shape}")
print(f"✓ Data normalized to range [0, 1]")

# ============================================
# STEP 3: BUILD CNN MODEL
# ============================================
print("\n[3] Building CNN Model...")

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    # Flatten and Dense Layers
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),
    
    # Output Layer
    layers.Dense(10, activation='softmax', name='output')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
model.summary()

# ============================================
# STEP 4: TRAIN THE MODEL WITH DATA AUGMENTATION
# ============================================
print("\n[4] Training Model with Data Augmentation...")
print("This may take 10-15 minutes depending on your hardware...")

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
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = augmented_model.fit(
    X_train, y_train_encoded,
    batch_size=128,
    epochs=25,
    validation_split=0.2,
    verbose=1
)

# Copy weights back to original model for saving
# (We save the model without augmentation layers)

print("\n✓ Training completed!")

# ============================================
# STEP 5: EVALUATE THE MODEL
# ============================================
print("\n[5] Evaluating Model on Test Set...")

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
# STEP 6: SAVE THE MODEL
# ============================================
print("\n[6] Saving Model...")

model.save('model/digit_model.h5')
print("✓ Model saved as 'model/digit_model.h5'")

# ============================================
# STEP 7: VISUALIZE RESULTS
# ============================================
print("\n[7] Creating Visualizations...")

# Plot 1: Training History
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axhline(y=0.985, color='g', linestyle='--', label='Target (98.5%)')
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
plt.savefig('model/training_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model/training_results.png'")

# Plot 2: Sample Predictions
plt.figure(figsize=(15, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    
    true_label = y_test[i]
    pred_label = y_pred_classes[i]
    confidence = np.max(y_pred[i]) * 100
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'T:{true_label} P:{pred_label}\n{confidence:.1f}%', 
              color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('model/sample_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Sample predictions saved as 'model/sample_predictions.png'")

# ============================================
# STEP 8: SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"✓ Model saved: model/digit_model.h5")
print(f"✓ Training results: model/training_results.png")
print(f"✓ Sample predictions: model/sample_predictions.png")
print(f"✓ Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nYou can now use this model in your Flask backend!")
print("=" * 60)

plt.show()