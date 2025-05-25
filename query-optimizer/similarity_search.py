import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from config import Config

class SimilaritySearch:
    def __init__(self):
        self.config = Config()
    
    def compute_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]
    
    def find_similar_examples(self, 
                            query_embedding: List[float], 
                            df: pd.DataFrame, 
                            k: int = None) -> pd.DataFrame:
        """Find K most similar examples for a given query embedding"""
        if k is None:
            k = self.config.SIMILARITY_K
        
        # Compute similarity with each row in the DataFrame
        similarities = []
        for _, row in df.iterrows():
            similarity = self.compute_cosine_similarity(query_embedding, row['embedding'])
            similarities.append(similarity)
        
        df_copy = df.copy()
        df_copy['similarity'] = similarities
        
        # Sort by similarity score (descending order)
        df_sorted = df_copy.sort_values(by='similarity', ascending=False)
        
        # Return top K matches
        top_matches = df_sorted.head(k)
        return top_matches
    
    def get_matched_conversations(self, 
                                query: str, 
                                query_embedding: List[float], 
                                df: pd.DataFrame, 
                                k: int = None) -> pd.DataFrame:
        """Get similar conversations for a given query"""
        similar_examples = self.find_similar_examples(query_embedding, df, k)
        
        return similar_examples[['conv_A_user', 'conv_A_assistant', 'conv_A_rating', 'similarity']]
