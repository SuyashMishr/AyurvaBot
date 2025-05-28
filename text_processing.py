"""
Module for text processing and analysis functions.
"""

import re
import string
from collections import Counter

def clean_text(text):
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def tokenize_text(text):
    """
    Split text into tokens (words).
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    tokens = text.split()
    
    return tokens

def get_word_frequencies(text):
    """
    Get word frequency counts from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Counter: Counter object with word frequencies
    """
    tokens = tokenize_text(text)
    return Counter(tokens)

def get_top_words(text, n=10):
    """
    Get the top N most frequent words in the text.
    
    Args:
        text (str): Input text
        n (int): Number of top words to return
        
    Returns:
        list: List of (word, count) tuples
    """
    word_freq = get_word_frequencies(text)
    return word_freq.most_common(n)

def extract_sentences(text):
    """
    Split text into sentences.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    # Simple sentence splitting (not perfect but works for basic cases)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
