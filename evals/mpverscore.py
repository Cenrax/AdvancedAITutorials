import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

class MoverScoreEvaluator:
    def __init__(self, api_key=None):
        """
        Initialize the MoverScore evaluator with GPT-4o API access.
        
        Args:
            api_key (str, optional): OpenAI API key for accessing GPT-4o. If None, loads from environment.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def embedding_with_backoff(self, **kwargs):
        """Wrapper for embeddings API with exponential backoff retry logic"""
        return self.client.embeddings.create(**kwargs)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, **kwargs):
        """Wrapper for chat completions API with exponential backoff retry logic"""
        return self.client.chat.completions.create(**kwargs)
    
    def get_embeddings_from_gpt4o(self, texts):
        """
        Get text embeddings using GPT-4o API with retry logic.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            list: List of embedding vectors
        """
        try:
            # Use text-embedding-3-large for higher quality embeddings
            embeddings_list = []
            
            for text in texts:
                response = self.embedding_with_backoff(
                    model="text-embedding-3-large",
                    input=text,
                    dimensions=1536  # High dimensionality for better semantic capture
                )
                
                embeddings_list.append(np.array(response.data[0].embedding))
            
            return embeddings_list
            
        except Exception as e:
            print(f"Error getting embeddings from OpenAI API: {e}")
            # Fall back to local model
            return self.get_local_embeddings(texts)
    
    def get_local_embeddings(self, texts):
        """
        Get text embeddings using local GPT-2 model as fallback.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            list: List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the last hidden state's mean as the embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
            
        return embeddings
    
    def calculate_word_importance(self, text):
        """
        Calculate importance weights for words in text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary mapping words to importance scores
        """
        # Simple TF-IDF like approach
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Calculate importance based on inverse frequency
        total_words = len(words)
        word_importance = {}
        
        for word, count in word_counts.items():
            # Words that appear less frequently are more important
            word_importance[word] = 1.0 / (count / total_words)
            
        return word_importance
    
    def calculate_moverscore(self, hypothesis, reference):
        """
        Calculate MoverScore between hypothesis and reference texts.
        
        Args:
            hypothesis (str): Generated text to evaluate
            reference (str): Ground truth text
            
        Returns:
            float: MoverScore value (higher means more similar)
        """
        # Get word-level embeddings
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        
        if not hyp_words or not ref_words:
            return 0.0
        # Get embeddings
        embeddings = self.get_embeddings_from_gpt4o([hypothesis, reference])
        if len(embeddings) < 2:
            # Fallback: use word-by-word cosine distance if embeddings fail
            return self.calculate_fallback_score(hyp_words, ref_words)
        
        hyp_embedding, ref_embedding = embeddings
        
        # Calculate Earth Mover's Distance (using Wasserstein)
        # Normalize embeddings for better results
        hyp_embedding_norm = hyp_embedding / np.linalg.norm(hyp_embedding)
        ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
        
        # Calculate Wasserstein distance between distributions
        distance = wasserstein_distance(hyp_embedding_norm, ref_embedding_norm)
        
        # Convert distance to similarity score (1 - normalized distance)
        # Lower distance = higher similarity
        score = 1.0 - min(distance, 1.0)
        
        return score
    
    def calculate_fallback_score(self, hyp_words, ref_words):
        """
        Calculate a fallback similarity score when embeddings aren't available.
        
        Args:
            hyp_words (list): Words in hypothesis
            ref_words (list): Words in reference
            
        Returns:
            float: Similarity score
        """
        # Simple Jaccard similarity as fallback
        hyp_set = set(hyp_words)
        ref_set = set(ref_words)
        
        intersection = len(hyp_set.intersection(ref_set))
        union = len(hyp_set.union(ref_set))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def evaluate_batch(self, hypotheses, references):
        """
        Evaluate a batch of hypotheses against references.
        
        Args:
            hypotheses (list): List of generated texts
            references (list): List of ground truth texts
            
        Returns:
            list: List of MoverScore values
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")
            
        scores = []
        for hyp, ref in zip(hypotheses, references):
            score = self.calculate_moverscore(hyp, ref)
            scores.append(score)
            
        return scores


if __name__ == "__main__":
    # Mock data - customer feedback summaries
    human_summaries = [
        "The user interface is confusing and difficult to navigate, especially on mobile devices.",
        "Customer service was responsive but unable to resolve my billing issue completely.",
        "The product exceeded expectations with its durability and high-quality materials."
    ]
    
    ai_generated_summaries = [
        "Users find the interface challenging to use, particularly when accessing via smartphones.",
        "Support team replied quickly but didn't fully fix the payment problem.",
        "The item surpassed anticipated quality levels with excellent construction and premium components."
    ]
    
    # Initialize evaluator (API key loaded from .env file)
    evaluator = MoverScoreEvaluator()
    
    # Calculate scores
    scores = evaluator.evaluate_batch(ai_generated_summaries, human_summaries)
    
    # Print results
    print("MoverScore Evaluation Results:")
    for i, (human, ai, score) in enumerate(zip(human_summaries, ai_generated_summaries, scores)):
        print(f"\nExample {i+1}:")
        print(f"Human: {human}")
        print(f"AI: {ai}")
        print(f"MoverScore: {score:.4f}")
        
    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage MoverScore: {avg_score:.4f}")
