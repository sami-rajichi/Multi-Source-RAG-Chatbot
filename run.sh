#!/usr/bin/env bash

# Enhanced RAG Chatbot Setup and Execution Script (No Virtualenv, pyenv removed)
# Installs Python 3.12.9 globally if needed
# Usage: ./run.sh [--install | --run | --docker | --clean | --help]

set -euo pipefail
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REQUIREMENTS="requirements.txt"

# Print usage help
show_help() {
    cat << EOF
${BLUE}Usage:${NC} $0 [--install | --run | --docker | --clean | --help]

Options:
  --install    Install Python and dependencies
  --run        Start the RAG Chatbot
  --docker     Build and start via Docker Compose
  --clean      Remove caches and temporary files
  --help       Show this help message
EOF
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == linux-* ]]; then
        command -v lsb_release &>/dev/null && lsb_release -is || echo Linux
    elif [[ "$OSTYPE" == darwin* ]]; then
        echo macOS
    else
        echo "$OSTYPE"
    fi
}

# Install dependencies globally or per-user
install_dependencies() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    if [[ "$OSTYPE" == linux-* ]]; then
        python -m pip install --user --upgrade pip
        python -m pip install --user -r "$REQUIREMENTS"
    else
        python -m pip install --upgrade pip
        python -m pip install -r "$REQUIREMENTS"
    fi
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Run the Streamlit application
run_app() {
    echo -e "${BLUE}Starting RAG Chatbot via Streamlit...${NC}"
    exec python -m streamlit run src/app.py
}

# Build and start via Docker Compose
run_docker() {
    echo -e "${YELLOW}Building and starting Docker containers...${NC}"
    docker-compose up --build --remove-orphans
}

# Clean up caches and temporary files
clean_up() {
    echo -e "${YELLOW}Cleaning up caches and Python artifacts...${NC}"
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
    find . -type f -name "*.py[co]" -delete
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Main
case "${1:-}" in
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
    --help)
        show_help
        ;;
    *)
        echo -e "${YELLOW}No argument provided. Installing and running...${NC}"
        install_dependencies
        run_app
        ;;
esac

exit 0
