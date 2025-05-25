import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any
import json
import os
from config import Config

class Utils:
    def __init__(self):
        self.config = Config()
    
    @staticmethod
    def calculate_statistics(optimized_scores: List[int], baseline_scores: List[int]) -> Dict[str, float]:
        """Calculate statistical metrics for the evaluation"""
        
        optimized_mean = np.mean(optimized_scores)
        baseline_mean = np.mean(baseline_scores)
        improvement = optimized_mean - baseline_mean
        improvement_percent = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(optimized_scores, baseline_scores)
        
        return {
            'optimized_mean': optimized_mean,
            'baseline_mean': baseline_mean,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    def save_results(self, results: List[Dict[str, Any]], stats: Dict[str, float], filename: str) -> None:
        """Save evaluation results to file"""
        
        output = {
            'statistics': stats,
            'detailed_results': results,
            'config': {
                'training_sample_size': self.config.TRAINING_SAMPLE_SIZE,
                'test_sample_size': self.config.TEST_SAMPLE_SIZE,
                'similarity_k': self.config.SIMILARITY_K,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'response_model': self.config.RESPONSE_MODEL
            }
        }
        
        filepath = os.path.join(self.config.RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def print_statistics(stats: Dict[str, float]) -> None:
        """Print statistical results in a formatted way"""
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Optimized Mean Score: {stats['optimized_mean']:.4f}")
        print(f"Baseline Mean Score: {stats['baseline_mean']:.4f}")
        print(f"Absolute Improvement: {stats['improvement']:.4f}")
        print(f"Percentage Improvement: {stats['improvement_percent']:.2f}%")
        print(f"T-statistic: {stats['t_statistic']:.4f}")
        print(f"P-value: {stats['p_value']:.6f}")
        print(f"Statistically Significant: {'Yes' if stats['is_significant'] else 'No'}")
        print("="*50)