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
    # Read image with high quality processing
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image file")
    
    # Preserve details while smoothing
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Enhanced edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 9)
    
    # Color quantization with more vibrant colors
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 12  # More colors for better gradients
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized = res.reshape((img.shape))
    
    # Sharpen the quantized image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    quantized = cv2.filter2D(quantized, -1, kernel)
    
    # Combine with edges for anime effect
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(quantized, 255 - edges)
    
    # Final quality enhancements
    result = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
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

        # Resize if needed while preserving aspect ratio
        with Image.open(temp_path) as img:
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.LANCZOS)  # High-quality resampling
                img.save(temp_path, quality=95, subsampling=0)  # Maximum quality
                print("Resized image with high quality settings")
            
        # Apply enhanced anime style effect
        result = apply_anime_style(temp_path)
        
        # Save result with maximum quality
        output_filename = 'anime_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        
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