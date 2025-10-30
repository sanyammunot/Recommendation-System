import pickle
import pandas as pd

class ModelManager:
    """Handles saving and loading of models and data"""
    
    @staticmethod
    def save_model(model, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
    @staticmethod
    def load_model(filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_data(data: pd.DataFrame, filepath: str):
        """Save data to disk"""
        data.to_csv(filepath, index=False)
        
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load data from disk"""
        return pd.read_csv(filepath)
