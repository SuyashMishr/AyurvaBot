"""
Configuration settings for the Ayurvedic data processing application.
"""

import os

# Define paths
# Dataset path
DATASET_PATH = "/Users/suyashmacair/Downloads/Ayurveda Dataset"

# Output directory for processed data
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Other configuration settings
DEBUG = True  # Set to False in production
SAMPLE_TEXT_LENGTH = 500  # Number of characters to show in sample text
