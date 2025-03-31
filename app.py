from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

def apply_anime_style(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Color quantization
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized = res.reshape((img.shape))
    
    # Combine edges with quantized colors
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(quantized, 255 - edges)
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(image.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(temp_path)

        file_size = os.path.getsize(temp_path)
        print(f"Processing file: {filename} ({file_size/1024:.1f}KB)")

        # Resize if needed
        with Image.open(temp_path) as img:
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))
                img.save(temp_path, quality=85)
                print("Resized image for processing")
            
        # Apply anime style effect
        result = apply_anime_style(temp_path)
        
        # Save result
        output_filename = 'anime_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, result)
        
        return jsonify({
            'url': f'/static/uploads/{output_filename}',
            'original': f'/static/uploads/{filename}'
        })
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, port=8000)