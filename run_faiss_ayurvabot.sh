#!/bin/bash

# Run script for Enhanced AyurvaBot with FAISS Vector Search and Hugging Face Document QA

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Enhanced AyurvaBot with FAISS Vector Search + Document QA ===${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}! Virtual environment not found.${NC}"
    echo "Running setup script first..."
    ./setup.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Setup failed. Please fix the issues and try again.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to activate virtual environment.${NC}"
    echo "Please run the setup script first: ./setup.sh"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install/upgrade required dependencies
echo -e "\n${GREEN}Installing/upgrading dependencies...${NC}"
pip install transformers torch gradio faiss-cpu sentence-transformers pillow --upgrade
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}! Some packages may have failed to install. Continuing...${NC}"
fi

# Create output directory if it doesn't exist
mkdir -p output

# Display information about the enhanced features
echo -e "\n${PURPLE}=== Enhanced AyurvaBot Features ===${NC}"
echo -e "${GREEN}✓${NC} Hugging Face Document QA Model (tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa)"
echo -e "${GREEN}✓${NC} FAISS Vector Search for Semantic Similarity"
echo -e "${GREEN}✓${NC} Sentence Transformers for Embeddings"
echo -e "${GREEN}✓${NC} Enhanced Content Filtering"
echo -e "${GREEN}✓${NC} AI + Traditional Knowledge Combination"
echo -e "${GREEN}✓${NC} User Feedback & Rating System"
echo -e "${GREEN}✓${NC} Professional Interface Design"
echo -e "${GREEN}✓${NC} Multi-Context Analysis"

# Display technical specifications
echo -e "\n${CYAN}=== Technical Specifications ===${NC}"
echo -e "${BLUE}•${NC} Document QA Model: tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa"
echo -e "${BLUE}•${NC} Vector Search: FAISS with Inner Product similarity"
echo -e "${BLUE}•${NC} Embeddings: Sentence Transformers (all-MiniLM-L6-v2)"
echo -e "${BLUE}•${NC} Content Filtering: Advanced Ayurvedic content prioritization"
echo -e "${BLUE}•${NC} Hardware: Optimized for Apple Silicon MPS"

# Run the Enhanced AyurvaBot with FAISS
echo -e "\n${GREEN}Starting Enhanced AyurvaBot with FAISS Vector Search...${NC}"
echo "The interface will be available at http://127.0.0.1:7860"
echo "Loading AI models and processing your Ayurveda dataset..."
echo "This may take a few minutes for initial setup..."
echo "Press Ctrl+C to stop the server."
echo ""
python3 enhanced_ayurvabot_faiss.py

# Deactivate virtual environment when done
deactivate

echo -e "${GREEN}Done!${NC}"
