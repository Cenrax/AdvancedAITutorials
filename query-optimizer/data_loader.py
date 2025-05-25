import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple
import pickle
import os
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        self.dataset = None
        self.training_data = None
        self.test_data = None
    
    def load_dataset(self) -> None:
        """Load the Unified Feedback Dataset from Hugging Face"""
        print("Loading dataset from Hugging Face...")
        try:
            self.dataset = load_dataset(self.config.DATASET_NAME)
            print(f"Dataset loaded successfully. Total samples: {len(self.dataset['train'])}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample training and test data from the dataset"""
        if self.dataset is None:
            self.load_dataset()
        
        print(f"Sampling {self.config.TRAINING_SAMPLE_SIZE} training samples...")
        print(f"Sampling {self.config.TEST_SAMPLE_SIZE} test samples...")
        
        # Convert to pandas DataFrame for easier manipulation
        df_full = pd.DataFrame(self.dataset['train'])
        
        # Sample data
        df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.training_data = df_shuffled.head(self.config.TRAINING_SAMPLE_SIZE)
        self.test_data = df_shuffled.tail(self.config.TEST_SAMPLE_SIZE)
        
        print(f"Training data shape: {self.training_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.training_data, self.test_data
    
    def save_data(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Save processed data to files"""
        training_path = os.path.join(self.config.DATA_DIR, "training_data.pkl")
        test_path = os.path.join(self.config.DATA_DIR, "test_data.pkl")
        
        with open(training_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"Data saved to {training_path} and {test_path}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data from files"""
        training_path = os.path.join(self.config.DATA_DIR, "training_data.pkl")
        test_path = os.path.join(self.config.DATA_DIR, "test_data.pkl")
        
        if os.path.exists(training_path) and os.path.exists(test_path):
            with open(training_path, 'rb') as f:
                training_data = pickle.load(f)
            
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            
            print("Data loaded from saved files")
            return training_data, test_data
        else:
            print("No saved data found. Loading fresh data...")
            return self.sample_data()