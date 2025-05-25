# ====================
# config.py
# ====================
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-ada-003"  # ada-3
    RESPONSE_MODEL = "gpt-4.1-nano"
    JUDGE_MODEL = "gpt-4.1-mini"
    
    # Algorithm Parameters
    TRAINING_SAMPLE_SIZE = 6000
    TEST_SAMPLE_SIZE = 100
    SIMILARITY_K = 10
    NUM_TRIALS = 20
    
    # Response Parameters
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    JUDGE_TEMPERATURE = 0.1
    
    # Dataset Configuration
    DATASET_NAME = "llm-blender/Unified-Feedback"
    
    # File Paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)