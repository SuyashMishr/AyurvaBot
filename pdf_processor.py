"""
Module for handling PDF processing operations.
"""

import os
import PyPDF2
import numpy as np
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return ""

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(reader.pages)

            print(f"Extracting text from PDF with {total_pages} pages...")

            for i, page in enumerate(reader.pages):
                if i % 10 == 0 and i > 0:
                    print(f"Processed {i}/{total_pages} pages...")
                page_text = page.extract_text()
                text += page_text + "\n"

        print(f"Text extraction complete. Extracted {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def save_text_to_file(text, output_path):
    """
    Save extracted text to a file.

    Args:
        text (str): Text to save
        output_path (str): Path to save the text file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {output_path}")
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}")

def analyze_text(text):
    """
    Perform basic analysis on the extracted text.

    Args:
        text (str): Extracted text from PDF

    Returns:
        dict: Dictionary containing analysis results
    """
    if not text:
        return {"word_count": 0, "line_count": 0, "char_count": 0}

    lines = text.split("\n")
    words = text.split()

    analysis = {
        "word_count": len(words),
        "line_count": len(lines),
        "char_count": len(text),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }

    return analysis

def create_text_dataframe(text):
    """
    Create a pandas DataFrame from the extracted text.

    Args:
        text (str): Extracted text from PDF

    Returns:
        pd.DataFrame: DataFrame containing text data
    """
    lines = text.split("\n")

    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Create a simple DataFrame
    df = pd.DataFrame({
        "line_number": range(1, len(lines) + 1),
        "text": lines,
        "char_count": [len(line) for line in lines],
        "word_count": [len(line.split()) for line in lines]
    })

    return df
