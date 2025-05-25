import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm

from config import Config
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from similarity_search import SimilaritySearch
from prompt_optimizer import PromptOptimizer
from response_generator import ResponseGenerator
from evaluator import Evaluator
from utils import Utils

class QueryOptimizer:
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.embedding_generator = EmbeddingGenerator()
        self.similarity_search = SimilaritySearch()
        self.prompt_optimizer = PromptOptimizer()
        self.response_generator = ResponseGenerator()
        self.evaluator = Evaluator()
        self.utils = Utils()
        
        # Data storage
        self.training_data = None
        self.test_data = None
        self.training_embeddings = None
    
    def prepare_data(self) -> None:
        """Prepare training and test data with embeddings"""
        
        print("Step 1: Loading and preparing data...")
        
        # Load or sample data
        self.training_data, self.test_data = self.data_loader.load_data()
        
        # Generate or load embeddings
        embeddings_file = "training_embeddings.pkl"
        
        try:
            self.training_embeddings = self.embedding_generator.load_embeddings(embeddings_file)
        except FileNotFoundError:
            print("Generating new embeddings...")
            queries = self.training_data['conv_A_user'].tolist()
            self.training_embeddings = self.embedding_generator.get_embeddings_batch(queries)
            self.embedding_generator.save_embeddings(self.training_embeddings, embeddings_file)
        
        # Add embeddings to training data
        self.training_data['embedding'] = self.training_embeddings
        
        print(f"Data preparation complete. Training samples: {len(self.training_data)}")
    
    def optimize_single_query(self, user_query: str) -> Dict[str, Any]:
        """Optimize response for a single query"""
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.get_embedding(user_query)
        
        # Find similar examples
        similar_examples = self.similarity_search.get_matched_conversations(
            user_query, query_embedding, self.training_data
        )
        
        # Generate optimized prompt
        optimization_result = self.prompt_optimizer.get_optimized_prompt(
            user_query, similar_examples
        )
        
        # Generate both responses
        responses = self.response_generator.generate_both_responses(
            user_query, optimization_result['optimized_prompt']
        )
        
        # Evaluate responses
        evaluation = self.evaluator.evaluate_responses(
            user_query,
            responses['optimized_response'],
            responses['baseline_response'],
            similar_examples
        )
        
        return {
            'query': user_query,
            'optimized_prompt': optimization_result['optimized_prompt'],
            'optimization_reasoning': optimization_result['reasoning'],
            'optimized_response': responses['optimized_response'],
            'baseline_response': responses['baseline_response'],
            'optimized_score': evaluation['optimized_score'],
            'baseline_score': evaluation['baseline_score'],
            'optimized_reasoning': evaluation['optimized_reasoning'],
            'baseline_reasoning': evaluation['baseline_reasoning'],
            'similar_examples_count': len(similar_examples)
        }
    
    def run_evaluation(self) -> None:
        """Run complete evaluation on test dataset"""
        
        print("\nStep 2: Running evaluation on test dataset...")
        
        results = []
        test_queries = self.test_data['conv_A_user'].tolist()
        
        # Process each test query
        for query in tqdm(test_queries, desc="Processing queries"):
            try:
                result = self.optimize_single_query(query)
                results.append(result)
            except Exception as e:
                print(f"Error processing query '{query[:50]}...': {e}")
                continue
        
        # Calculate statistics
        optimized_scores = [r['optimized_score'] for r in results]
        baseline_scores = [r['baseline_score'] for r in results]
        
        stats = self.utils.calculate_statistics(optimized_scores, baseline_scores)
        
        # Print results
        self.utils.print_statistics(stats)
        
        # Save results
        self.utils.save_results(results, stats, "evaluation_results.json")
        
        return results, stats
    
    def run_demo(self, demo_queries: List[str] = None) -> None:
        """Run demo with sample queries"""
        
        if demo_queries is None:
            demo_queries = [
                "How do I reset my password?",
                "What is the company's vacation policy?",
                "How to improve team productivity?",
                "Explain machine learning in simple terms",
                "What are the best practices for remote work?"
            ]
        
        print("\nStep 3: Running demo with sample queries...")
        
        for query in demo_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            try:
                result = self.optimize_single_query(query)
                
                print(f"Optimized Response Score: {result['optimized_score']}")
                print(f"Baseline Response Score: {result['baseline_score']}")
                print(f"\nOptimized Response:\n{result['optimized_response']}")
                print(f"\nBaseline Response:\n{result['baseline_response']}")
                
            except Exception as e:
                print(f"Error processing query: {e}")

def main():
    """Main execution function"""
    
    print("OpenAI Query Optimizer - Starting...")
    
    # Initialize optimizer
    optimizer = QueryOptimizer()
    
    # Prepare data
    optimizer.prepare_data()
    
    # Run evaluation
    results, stats = optimizer.run_evaluation()
    
    # Run demo
    optimizer.run_demo()
    
    print("\nOptimization complete!")