#!/bin/bash

# Setup script for AyurvaBot
# This script will set up the environment and install all dependencies

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AyurvaBot Setup ===${NC}"
echo "This script will set up everything needed to run AyurvaBot."

# Check if Python 3 is installed
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Found $PYTHON_VERSION"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if pip is installed
if command -v pip3 &>/dev/null; then
    PIP_VERSION=$(pip3 --version | awk '{print $1 " " $2}')
    echo -e "${GREEN}✓${NC} Found $PIP_VERSION"
else
    echo -e "${YELLOW}! pip3 not found. Attempting to install pip...${NC}"
    python3 -m ensurepip --upgrade
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install pip. Please install pip manually and try again.${NC}"
        exit 1
    fi
fi

# Create and activate virtual environment
echo -e "\n${GREEN}=== Setting up virtual environment ===${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}! Virtual environment already exists. Reusing it.${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to create virtual environment. Please install venv manually:${NC}"
        echo "python3 -m pip install virtualenv"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install dependencies
echo -e "\n${GREEN}=== Installing dependencies ===${NC}"
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to install dependencies.${NC}"
    echo "Attempting to install dependencies individually..."
    
    # Try installing each dependency individually
    for package in numpy pandas PyPDF2 matplotlib; do
        echo "Installing $package..."
        pip install $package
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Failed to install $package.${NC}"
        else
            echo -e "${GREEN}✓${NC} Installed $package"
        fi
    done
else
    echo -e "${GREEN}✓${NC} All dependencies installed successfully"
fi

# Check if the PDF file exists
echo -e "\n${GREEN}=== Checking PDF file ===${NC}"
PDF_PATH="/Users/suyashmacair/Downloads/Downloads_17012023_Ayurvedic Standard Treatment Guildelines-4.pdf"
if [ -f "$PDF_PATH" ]; then
    echo -e "${GREEN}✓${NC} Found PDF file at $PDF_PATH"
else
    echo -e "${YELLOW}! PDF file not found at $PDF_PATH${NC}"
    echo "Please update the PDF path in config.py before running the application."
fi

# Create output directory
echo -e "\n${GREEN}=== Setting up directories ===${NC}"
mkdir -p output
echo -e "${GREEN}✓${NC} Created output directory"

# Make run.sh executable
chmod +x run.sh
echo -e "${GREEN}✓${NC} Made run.sh executable"

echo -e "\n${GREEN}=== Setup complete! ===${NC}"
echo -e "You can now run the application with: ${YELLOW}./run.sh${NC}"
echo -e "Or activate the virtual environment and run: ${YELLOW}python3 main.py${NC}"

# Ask if user wants to run the application now
echo -e "\nDo you want to run the application now? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "\n${GREEN}=== Running AyurvaBot ===${NC}"
    python3 main.py
else
    echo -e "\nYou can run the application later with: ${YELLOW}./run.sh${NC}"
fi

# Deactivate virtual environment
deactivate
