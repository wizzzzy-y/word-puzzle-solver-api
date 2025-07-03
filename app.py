import os
import logging
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
from word_solver import WordPuzzleSolver
from high_performance_solver import solve_word_puzzle

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize word puzzle solver
solver = WordPuzzleSolver()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_puzzle():
    """Main endpoint for solving word puzzles"""
    try:
        # Check if file was uploaded
        if 'screenshot' not in request.files:
            return jsonify({'error': 'No screenshot file provided'}), 400
        
        file = request.files['screenshot']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filepath}")
        
        # Use high-performance solver for optimal speed
        result = solve_word_puzzle(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing puzzle: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/test', methods=['GET', 'POST'])
def test_upload():
    """Test page for uploading and displaying results"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'screenshot' not in request.files:
                flash('No screenshot file provided', 'error')
                return redirect(request.url)
            
            file = request.files['screenshot']
            
            # Check if file is selected
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check if file is allowed
            if not allowed_file(file.filename):
                flash('File type not allowed', 'error')
                return redirect(request.url)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Processing test image: {filepath}")
            
            # Use high-performance solver for optimal speed
            result = solve_word_puzzle(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return render_template('index.html', result=result, result_json=json.dumps(result, indent=2))
            
        except Exception as e:
            logger.error(f"Error processing test puzzle: {str(e)}")
            flash(f'Processing failed: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
