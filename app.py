"""
Handwritten Digit Recognition - Flask Backend API
Author: Htoo Aunt
Description: REST API to serve the trained digit recognition model
"""


from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import keras
from PIL import Image
import io
import os
from tensorflow._api.v2.raw_ops import Imag

app = Flask(__name__)
CORS(app)

# Model configuration
MODEL_PATH = 'model/digit_model.h5'
MODEL_DESCRIPTION = 'Combined MNIST + Custom handwriting data'

# Current model state
current_model = None
model_modified = None

def load_model():
    """Load the digit recognition model"""
    global current_model, model_modified

    if not os.path.exists(MODEL_PATH):
        return False, f"Model file not found: {MODEL_PATH}"

    try:
        current_model = keras.models.load_model(MODEL_PATH)
        model_modified = os.path.getmtime(MODEL_PATH)
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

print("=" * 60)
print("LOADING DIGIT RECOGNITION MODEL...")
print("=" * 60)

# Load the model
if os.path.exists(MODEL_PATH):
    success, message = load_model()
    if success:
        print(f"✓ {message}")
        print(f"  ({MODEL_DESCRIPTION})")
    else:
        print(f"❌ {message}")
else:
    print(f"❌ No model file found at {MODEL_PATH}")
    print("Please train a model first using train_model.py or train_custom_model.py")

print("=" * 60)


def preprocess_image(image_data):
    """
    Preprocess the input image for model prediction

    Args:
        image_data: Base64 encoded image string

    Returns:
        Preprocessed numpy array ready for prediction
    """
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        image_array = np.array(image)

        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA
                # For canvas: use alpha channel or RGB
                # If alpha exists, use it as the drawing mask
                alpha = image_array[:, :, 3]
                rgb = image_array[:, :, :3]
                # Convert RGB to grayscale
                gray = np.mean(rgb, axis=2)
                # Use alpha to determine where drawing is
                if np.max(alpha) > 0:
                    # Canvas typically has drawing in RGB with alpha
                    image_array = gray
                else:
                    image_array = gray
            elif image_array.shape[2] == 3:  # RGB
                image_array = np.mean(image_array, axis=2)

        # Invert if background is white (digit should be white on black)
        # MNIST expects white digit on black background
        if np.mean(image_array) > 127:
            image_array = 255 - image_array

        # Convert to uint8 for OpenCV operations
        image_array = image_array.astype(np.uint8)

        # Find bounding box of the digit and center it (like MNIST preprocessing)
        coords = cv2.findNonZero(image_array)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Extract the digit region
            digit = image_array[y:y+h, x:x+w]

            # Create a square canvas with padding
            max_dim = max(w, h)
            # Add 20% padding around the digit
            padding = int(max_dim * 0.2)
            canvas_size = max_dim + 2 * padding

            # Create black canvas
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

            # Center the digit on the canvas
            x_offset = (canvas_size - w) // 2
            y_offset = (canvas_size - h) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit

            image_array = canvas

        # Adaptive stroke thickness enhancement
        # Detect if strokes are thin and need thickening
        stroke_pixels = np.sum(image_array > 0)
        total_pixels = image_array.size
        stroke_density = stroke_pixels / total_pixels

        # Calculate average stroke intensity
        if stroke_pixels > 0:
            avg_stroke_intensity = np.mean(image_array[image_array > 0])
        else:
            avg_stroke_intensity = 0

        # Apply dilation only if strokes are thin (low density or low intensity)
        # Thin strokes typically have density < 0.15 or low average intensity
        if stroke_density < 0.20 or avg_stroke_intensity < 180:
            # Thin strokes detected - apply dilation to preserve detail
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            image_array = cv2.dilate(image_array, kernel, iterations=1)

        # Resize to 28x28 using anti-aliasing
        image_pil = Image.fromarray(image_array)
        image_pil = image_pil.resize((28, 28), Image.LANCZOS)
        image_array = np.array(image_pil)

        # Apply slight Gaussian blur to smooth pixelation (like MNIST)
        image_array = cv2.GaussianBlur(image_array, (3, 3), 0)

        # Enhance contrast to ensure strokes are visible
        # Apply histogram normalization if the image is too faint
        if np.max(image_array) > 0:
            # Stretch histogram to use full range
            image_array = ((image_array - np.min(image_array)) /
                          (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)

        # Normalize to 0-1 range
        image_array = image_array.astype('float32') / 255.0

        # Reshape for model input
        image_array = image_array.reshape(1, 28, 28, 1)

        # Debug: Save preprocessed image to see what model receives
        debug_img = (image_array[0, :, :, 0] * 255).astype(np.uint8)
        cv2.imwrite('debug_preprocessed.png', debug_img)
        print(f"Debug: Image saved. Min={image_array.min():.3f}, Max={image_array.max():.3f}, Mean={image_array.mean():.3f}")

        return image_array

    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/',methods=['GET'])
def home():
    """Home route to check if the API is running"""
    return jsonify({
        'message': 'Handwritten Digit Recognition API',
        'version': '1.0',
        'author': 'Htoo Aunt',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'POST - Predict digit from image',
            '/model-info': 'GET - Model information'
        }
    })
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    if current_model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded',
        }),500

    import datetime
    modified_time = datetime.datetime.fromtimestamp(model_modified).strftime('%Y-%m-%d %H:%M:%S') if model_modified else 'Unknown'

    return jsonify({
        'success': True,
        'model_name': 'CNN for Digit Recognition',
        'model_path': MODEL_PATH,
        'trained_on': MODEL_DESCRIPTION,
        'last_modified': modified_time,
        'input_shape': [28, 28, 1],
        'output_classes': 10,
        'architecture': {
            'layers': [
                'Conv2D (32 filters, 3x3)',
                'MaxPooling2D (2x2)',
                'Conv2D (64 filters, 3x3)',
                'MaxPooling2D (2x2)',
                'Flatten',
                'Dense (128 units)',
                'Dropout (0.5)',
                'Dense (10 units, softmax)'
            ]
        }
    })

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Reload the model (useful after retraining)"""
    try:
        success, message = load_model()
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/predict', methods=['POST'])
def predict_digit():
    """
    Predict digit from uploaded image
    
    Expected JSON format:
    {
        "image": "data:image/png;base64,iVBORw0KGgo..."
    }
    
    Returns:
    {
        "success": true,
        "digit": 7,
        "confidence": 98.5,
        "all_probabilities": {
            "0": 0.1,
            "1": 0.2,
            ...
        }
    }
    """
    if current_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500

    try:
        data = request.json

        if 'image' not in data:
            return jsonify({
                 'success': False,
                 'error': 'No image data provided'
            }),400

        image_data = data ['image']

        processed_image = preprocess_image(image_data)

        prediction = current_model.predict(processed_image,verbose =0)


        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)

        all_probabilities = {
            str(i) :float (prediction[0][i] *100)
            for i in range(10)
        }
        print(f"Prediction: {predicted_digit} (Confidence: {confidence:.2f}%)")

        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        })
    except ValueError as  ve:
           return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
        """
    Predict multiple digits from multiple images
    
    Expected JSON format:
    {
        "images": ["data:image/png;base64,...", "data:image/png;base64,..."]
    }
    """
        if current_model is None:
            return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
        try:
            data = request.json

            if 'images' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No images data provided'
                }), 400

            images = data['images']
            results = []

            for idx, image_data in enumerate(images):
                try:
                    processed_image = preprocess_image(image_data)
                    predictions = current_model.predict(processed_image, verbose=0)

                    predicted_digit = int(np.argmax(predictions[0]))
                    confidence = float(np.max(predictions[0]) * 100)

                    results.append({
                        'index': idx,
                        'digit': predicted_digit,
                        'confidence': confidence
                    })
                except Exception as e:
                    results.append({
                        'index': idx,
                        'error': str(e)
                    })

            return jsonify({
                'success': True,
                'predictions': results
            })
        
        except Exception as e:
            return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING FLASK SERVER...")
    print("=" * 60)
    print("Server: http://localhost:5000")
    print("Health Check: http://localhost:5000/health")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
