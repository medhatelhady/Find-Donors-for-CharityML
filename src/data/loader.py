import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os


class DataLoader:
    """Load and validate data for ML pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.label = self.config['data']['label']
        
    def load_data(self, path=None):
        """Load data from CSV or generate sample data."""
        if path is None:
            path = self.config['data']['raw_path']
            
        if os.path.exists(path):
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
        else:
            print("Generating sample classification data...")
            df = self._generate_sample_data()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Sample data saved to {path}")
            
        return df
    
    
    def validate_data(self, df):
        """Basic data validation."""
        print(f"Data shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Target distribution:\n{df[self.label].value_counts()}")
        return True
