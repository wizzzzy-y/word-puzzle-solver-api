import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from solver import WordPuzzleSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "koyeb-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize solver
solver = WordPuzzleSolver()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({"status": "Word Puzzle Solver API is running", "version": "2.0"})

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "dictionary_size": len(solver.dictionary),
        "upload_folder": UPLOAD_FOLDER
    })

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

        # Process the image and solve the puzzle
        result = solver.solve_puzzle(filepath)

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing puzzle: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
