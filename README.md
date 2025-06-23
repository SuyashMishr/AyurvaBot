---
title: AyurvaBot - RAG-Powered Ayurvedic Assistant
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
# Enhanced AyurvaBot

An intelligent Ayurvedic medicine assistant using fine-tuned DistilBERT with FAISS vector search and web-enhanced RAG (Retrieval Augmented Generation).
<img width="1470" alt="Screenshot 2025-06-15 at 9 13 47 AM" src="https://github.com/user-attachments/assets/fdac4449-0886-4ad5-a7ce-5109c6a3f4ef" />
![Screenshot 2025-06-15 at 9 13 37 AM](https://github.com/user-attachments/assets/0d0b699b-5c09-4528-a066-595f3ae85d05)

<img width="1470" alt="Screenshot 2025-06-15 at 9 14 00 AM" src="https://github.com/user-attachments/assets/ca61fc71-5bd0-4c53-93c8-ee52da1d3d1c" />

## Features

- 🌿 **Fine-tuned DistilBERT Model**: Trained on comprehensive Ayurvedic texts for accurate responses
- 🔍 **Optimized FAISS Vector Search**: Fast and efficient semantic search capability
- 🌐 **Web-Enhanced RAG Pipeline**: Retrieves up-to-date information from the web
- 📚 **Smart Context Integration**: Combines knowledge from multiple sources
- 💾 **Persistent Index Storage**: Faster startup with saved indices
- 📊 **User Feedback System**: Continuous improvement through feedback

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Ayurveda Dataset (configured in `src/config.py`)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Bot

To start the AyurvaBot with default settings:

```bash
./run_ayurvabot.sh
```

The interface will be available at http://0.0.0.0:7860

### Fine-Tuning the Model

To fine-tune the model on the Ayurveda dataset:

**Method 1**: Using the fine-tuning script
```bash
python fine_tune_model.py
```

**Method 2**: When running the bot
```bash
./run_ayurvabot.sh --fine-tune
```

Fine-tuning may take some time depending on your hardware. The fine-tuned model will be saved to `output/fine_tuned_model/final`.

## Usage

1. Enter your question about Ayurveda in the input box
2. Click "Ask Question" to get a response
3. Toggle "Include web search results" to enable/disable retrieving information from the web
4. Rate responses and provide feedback to help improve the system

## File Structure

- `src/` - Source code for the AyurvaBot components
  - `config.py` - Configuration settings
  - `knowledge_base.py` - Ayurvedic knowledge base
  - `feedback_manager.py` - User feedback collection and storage
  - `web_search.py` - Web search component
  - `rag_pipeline.py` - RAG pipeline implementation
  - `model_fine_tuner.py` - Model fine-tuning functionality
- `working_ayurvabot.py` - Main application file
- `run_ayurvabot.sh` - Convenience script to run the application
- `fine_tune_model.py` - Script to fine-tune the model
- `output/` - Directory for processed data, FAISS indices, and fine-tuned models

## Configuration

Update the dataset path in `src/config.py` to point to your Ayurveda dataset:

```python
self.DATASET_PATH = "/path/to/your/ayurveda_dataset"
```

## Example Questions

- "What is Ayurveda?"
- "What are the three doshas and their functions?"
- "How is fever treated in Ayurveda?"
- "What herbs are good for heart health?"
- "What is Panchakarma and its benefits?"
- "Which herbs boost immunity in Ayurveda?"
- "How does Ayurveda treat digestive problems?"
- "What is the latest research on Ayurvedic herbs?"
- "What is Shirodhara therapy?"
