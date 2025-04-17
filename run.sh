#!/bin/bash

# RAG Chatbot Setup and Execution Script with Virtual Environment
# Usage: ./run.sh [--install | --run | --docker | --clean]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VENV_NAME="rag-env"
VENV_DIR="./$VENV_NAME"
REQUIREMENTS="requirements.txt"

# Function to create virtual environment
create_venv() {
    echo -e "${BLUE}Checking Python version...${NC}"
    python3 --version || { echo -e "${RED}Python 3 is required${NC}"; exit 1; }
    
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
    else
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    if [ ! -d "$VENV_DIR" ]; then
        create_venv
    fi
    
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
    
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${RED}Failed to install dependencies.${NC}"
        exit 1
    fi
    
    deactivate
}

# Function to run the application
run_app() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}Virtual environment not found. Running installer first...${NC}"
        install_dependencies
    fi
    
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
    
    echo -e "${YELLOW}Starting RAG Chatbot...${NC}"
    streamlit run src/app.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start the application.${NC}"
        deactivate
        exit 1
    fi
    
    deactivate
}

# Function to run with Docker
run_docker() {
    echo -e "${YELLOW}Building and starting Docker containers...${NC}"
    docker-compose up --build
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start Docker containers.${NC}"
        exit 1
    fi
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -d "$VENV_DIR" ]; then
        echo -e "${BLUE}Removing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    fi
    
    echo -e "${BLUE}Cleaning Python cache...${NC}"
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type d -name ".pytest_cache" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Main script logic
case "$1" in
    --install)
        install_dependencies
        ;;
    --run)
        run_app
        ;;
    --docker)
        run_docker
        ;;
    --clean)
        clean_up
        ;;
    *)
        echo -e "${YELLOW}Starting full setup...${NC}"
        install_dependencies
        run_app
        ;;
esac

exit 0
