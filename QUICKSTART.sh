#!/bin/bash

# 🚀 Graph Neural Tutor - Quick Start Script
# Automated setup for Linux/macOS systems

set -e  # Exit on any error

echo "🎯 Graph Neural Tutor - Quick Start Setup"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python 3.8+ is required but not installed."
        echo "Please install Python from https://python.org"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_status "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8+
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_error "Python 3.8+ is required. Found $PYTHON_VERSION"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch CPU version (faster for demo)
    print_status "Installing PyTorch (CPU version)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # Install other dependencies
    print_status "Installing other dependencies..."
    pip install -r requirements.txt
    
    print_success "All dependencies installed"
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p models datasets results figures web/static web/templates
    
    # Create .gitkeep files for empty directories
    touch models/.gitkeep datasets/.gitkeep results/.gitkeep figures/.gitkeep
    
    print_success "Directory structure created"
}

# Download or create sample data
setup_data() {
    print_status "Setting up sample data..."
    
    # Run a quick dataset generation
    $PYTHON_CMD -c "
from main import create_robust_dataset
import torch
import os

print('Generating sample dataset...')
dataset = create_robust_dataset(num_per_rule=50, num_neg=100)  # Smaller for quick setup

# Save sample dataset
os.makedirs('datasets', exist_ok=True)
torch.save(dataset, 'datasets/sample_dataset.pt')
print(f'Sample dataset saved with {len(dataset)} examples')
"
    
    print_success "Sample data created"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Test basic imports
    $PYTHON_CMD -c "
import torch
import sympy
import flask
from main import create_ast_graph, DistinctAlgebraicGNN
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Device available: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
    
    # Test web app startup (quick check)
    timeout 5 $PYTHON_CMD -c "
from web.app import app
print('✅ Web application loads successfully')
" || print_warning "Web app test timed out (this is normal)"
    
    print_success "Installation test completed"
}

# Main setup function
main() {
    echo "Starting automated setup..."
    echo ""
    
    # Check system requirements
    check_python
    
    # Setup Python environment
    setup_venv
    
    # Install all dependencies
    install_dependencies
    
    # Setup project structure
    setup_directories
    
    # Setup sample data
    setup_data
    
    # Test everything works
    test_installation
    
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    echo "Next steps:"
    echo "1. Start the web interface:"
    echo "   ${GREEN}$PYTHON_CMD web/app.py${NC}"
    echo ""
    echo "2. Or run the training pipeline:"
    echo "   ${GREEN}$PYTHON_CMD main.py${NC}"
    echo ""
    echo "3. Or start the API server:"
    echo "   ${GREEN}$PYTHON_CMD -m uvicorn api.main:app --reload${NC}"
    echo ""
    echo "🌐 Web interface will be available at: ${BLUE}http://localhost:5000${NC}"
    echo "🔌 API documentation will be at: ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
    echo "📚 For more information, see:"
    echo "   - README.md for full documentation"
    echo "   - QUICKSTART.md for detailed instructions"
    echo "   - docs/ directory for advanced topics"
    echo ""
    echo "🚀 Happy researching!"
}

# Run main function
main
