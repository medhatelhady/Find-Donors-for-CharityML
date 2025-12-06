import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer


def load_config(config_path='configs/config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training pipeline."""
    print("="*60)
    print("ML PIPELINE - TRAINING")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Step 1: Load data
    print("\n[1/4] Loading data...")
    loader = DataLoader(config)
    df = loader.load_data()
    loader.validate_data(df)
    
    # Step 2: Prepare data and build preprocessing pipeline
    print("\n[2/4] Building preprocessing pipeline...")
    preprocessor = DataPreprocessor(config)
    
    # Separate features and target
    label = config['data']['label']
    X = df.drop(label, axis=1)
    y = df[label].map({'>50K': 1, '<=50K': 0})
    
    # Build preprocessing pipeline
    preprocessor.build_preprocessing_pipeline()
    
    # Split data (raw data, preprocessing will be done in pipeline)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 3: Train Decision Tree pipeline
    print("\n[3/4] Training Decision Tree pipeline...")
    trainer = ModelTrainer(config, preprocessor)
    trainer.train(X_train, y_train)
    
    # Step 4: Evaluate pipeline
    print("\n[4/4] Evaluating pipeline...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save pipeline
    pipeline_path = trainer.save_pipeline()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model: Decision Tree")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Pipeline saved to: {pipeline_path}")
    print("\nNext steps:")
    print("  1. Build: docker build -t ml-api .")
    print("  2. Run: docker run -p 5000:5000 ml-api")
    print("  3. Test: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @sample_request.json")
    print("="*60)


if __name__ == '__main__':
    main()
