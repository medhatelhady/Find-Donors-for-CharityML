from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import yaml
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = Flask(__name__)
CORS(app)

# Global variables
pipeline = None
config = None


def load_config():
    """Load configuration."""
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def initialize():
    """Initialize complete pipeline."""
    global pipeline, config
    
    config = load_config()
    
    # Load complete pipeline (preprocessing + model)
    pipeline_path = config['deployment']['model_path']
    pipeline = joblib.load(pipeline_path)
    print(f"Complete pipeline loaded from {pipeline_path}")
    
    print(f"\n{'='*50}")
    print("API READY!")
    print(f"{'='*50}")
    print(f"Listening on http://{config['deployment']['host']}:{config['deployment']['port']}")
    print(f"\nEndpoints:")
    print(f"  GET  /health  - Health check")
    print(f"  POST /predict - Make predictions")
    print(f"{'='*50}\n")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'pipeline_loaded': pipeline is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Get input data
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data['features']])
        
        # Predict using complete pipeline (preprocessing + model)
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0].tolist()
        
        response = {
            'prediction': int(prediction),
            'probability': probability
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    initialize()
    app.run(
        host=config['deployment']['host'],
        port=config['deployment']['port'],
        debug=False
    )
