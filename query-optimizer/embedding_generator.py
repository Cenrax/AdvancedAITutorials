import numpy as np
from openai import OpenAI
from typing import List, Union
import time
import pickle
import os
from config import Config
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI ada-3"""
        try:
            response = self.client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            time.sleep(1)  # Rate limiting
            return self.get_embedding(text)  # Retry
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches"""
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def save_embeddings(self, embeddings: List[List[float]], filename: str) -> None:
        """Save embeddings to file"""
        filepath = os.path.join(self.config.DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filename: str) -> List[List[float]]:
        """Load embeddings from file"""
        filepath = os.path.join(self.config.DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Embeddings loaded from {filepath}")
            return embeddings
        else:
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")