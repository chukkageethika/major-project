from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy classification function (replace with actual ML model later)
def classify_plastic(img):
    return "Dummy Plastic Type"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Simulate ML model processing
    prediction = classify_plastic(None)  # Dummy function
    return jsonify({'filename': filename, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
