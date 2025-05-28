# AyurvaBot

A Python application for processing and analyzing Ayurvedic text data from PDF files.

## Overview

This application extracts text from Ayurvedic PDF documents, performs text analysis, and generates visualizations and data files for further processing.

## Features

- PDF text extraction
- Text cleaning and preprocessing
- Basic text analysis (word count, sentence extraction, etc.)
- Word frequency analysis
- Data visualization
- Export to CSV and text files

## Project Structure

```
AyurvaBot/
├── config.py              # Configuration settings
├── main.py                # Main application script
├── pdf_processor.py       # PDF processing functions
├── text_processing.py     # Text analysis functions
├── output/                # Generated output files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Update the PDF file path in `config.py` if needed:
   ```python
   LOCAL_PDF_PATH = "/path/to/your/pdf/file.pdf"
   ```

## Usage

Run the main script:
```
python main.py
```

This will:
1. Extract text from the specified PDF file
2. Clean and analyze the text
3. Generate statistics and visualizations
4. Save the results to the output directory

## Output Files

The application generates several output files in the `output` directory:
- `extracted_text_[timestamp].txt`: Raw extracted text
- `text_data_[timestamp].csv`: Text data in CSV format
- `word_count_plot_[timestamp].png`: Visualization of word counts
