import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get tokens from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class Config:
    """
    Configuration settings for the AyurvaBot application.
    """
    def __init__(self):
        self.HF_TOKEN = HUGGINGFACE_TOKEN
        self.QA_MODEL = "distilbert-base-cased-distilled-squad"
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.PORT = int(os.getenv("PORT", 7860))
        self.TOP_K_RETRIEVAL = 7
        self.SIMILARITY_THRESHOLD = 0.3  # Lower to be more inclusive
        self.CONFIDENCE_THRESHOLD = 0.05  # Lower to get more answers
        self.MAX_CONTEXT_LENGTH = 1024   # Increase context length
        
        # Dataset path for Ayurveda dataset
        self.DATASET_PATH = "/Users/suyashmacair/Downloads/Ayurveda Dataset"
        
        # Output directory for processed data and models
        self.OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        
        # Ensure output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)