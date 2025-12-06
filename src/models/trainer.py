from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from datetime import datetime


class ModelTrainer:
    """Train Decision Tree classifier with complete pipeline."""
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.pipeline = None
        self.metrics = {}
    
    def build_pipeline(self):
        """Build complete pipeline: preprocessing + Decision Tree."""
        model = DecisionTreeClassifier(
            random_state=self.config['data']['random_state'],
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        )
        
        # Create pipeline with preprocessing and model
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor.pipeline),
            ('classifier', model)
        ])
        
        return self.pipeline
    
    def train(self, X_train, y_train):
        """Train the complete pipeline."""
        print(f"\n{'='*50}")
        print("Training Decision Tree pipeline...")
        print(f"{'='*50}")
        
        # Build pipeline
        self.build_pipeline()
        
        # Train
        self.pipeline.fit(X_train, y_train)
        
        print("✓ Training complete!")
        
        return self.pipeline
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained pipeline."""
        print(f"\n{'='*50}")
        print("Evaluation Results")
        print(f"{'='*50}")
        
        if self.pipeline is None:
            raise ValueError("No pipeline trained yet. Call train() first.")
        
        # Predict
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"\nDecision Tree:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  F1 Score:  {self.metrics['f1']:.4f}")
        
        print(f"\n{'='*50}")
        
        return self.metrics
    
    def save_pipeline(self):
        """Save the trained pipeline."""
        if self.pipeline is None:
            raise ValueError("No pipeline trained yet. Call train() first.")
        
        output_path = self.config['model']['output_path']
        os.makedirs(output_path, exist_ok=True)
        
        # Save complete pipeline (preprocessing + model)
        pipeline_path = os.path.join(output_path, 'best_model.pkl')
        joblib.dump(self.pipeline, pipeline_path)
        
        # Save metadata
        metadata = {
            'model_name': 'decision_tree',
            'accuracy': self.metrics.get('accuracy', 0),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(output_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPipeline saved to {pipeline_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return pipeline_path
