import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


# Define log transform function at module level (so it can be pickled)
def log_transform(x):
    """Apply log1p transformation."""
    return np.log1p(x)


class DataPreprocessor:
    """Preprocess data for ML pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.label = self.config['data']['label']
        
        # Define column groups
        self.skewed = ['capital-gain', 'capital-loss']
        self.numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.categorical = ["marital-status", "relationship", "race", "sex"]

        self.categorical_with_many_unique_values = ["education_level", "occupation", "native-country", "workclass"]
    def preprocess(self, df, fit=True):
        """Complete preprocessing pipeline using ColumnTransformer."""
        df = df.copy()
        
        # Separate features and target
        if self.label in df.columns:
            X = df.drop(self.label, axis=1)
            y = df[self.label]
        else:
            raise ValueError(f"Target column ({self.label}) does not exist")
        
        # Build pipeline if not exists
        if self.pipeline is None:
            self.build_preprocessing_pipeline()
        
        # Fit and transform or just transform
        if fit:
            X_transformed = self.pipeline.fit_transform(X)
        else:
            X_transformed = self.pipeline.transform(X)
        
        # Map target labels
        y = self._map_target(y)
        
        return X_transformed, y
    
    def build_preprocessing_pipeline(self):
        """Build complete preprocessing pipeline using ColumnTransformer."""
        
        # Get scaler based on config
        scaler = MinMaxScaler()
        binary_encoder = BinaryEncoder()
        
        # Create transformers for each step (use module-level function for pickling)
        log_transformer = FunctionTransformer(log_transform, validate=False)
        numerical_imputer = SimpleImputer(strategy="mean")
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="if_binary")
        

        numerical_pipeline = Pipeline([
                ('imputer', numerical_imputer),
                ('scaler', scaler)
            ])

        # Build ColumnTransformer with all steps
        self.pipeline = ColumnTransformer(
            transformers=[
                # Step 1: Log transform skewed columns
                ('log_skewed', log_transformer, self.skewed),
                # Step 2: Impute and scale numerical columns
                ('numerical', numerical_pipeline, self.numerical),
                # Step 3: Impute categorical columns
                #('categorical_imputer', categorical_imputer, self.categorical),
                # Step 4: encode categorical columns
                ("encode", encoder, self.categorical),
                ("binary_encoder", binary_encoder, self.categorical_with_many_unique_values)
            ],
            remainder='passthrough'
        )

        return self.pipeline
        
    def _map_target(self, y):
        """Map target labels to binary values."""
        return y.map({'>50K': 1, '<=50K': 0})

    def split_data(self, X, y):
        """Split data into train and test sets."""
        return train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
    
    def save(self, path):
        """Save preprocessor pipeline."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Preprocessor pipeline saved to {path}")
    
    def load(self, path):
        """Load preprocessor pipeline."""
        self.pipeline = joblib.load(path)
        print(f"Preprocessor pipeline loaded from {path}")
