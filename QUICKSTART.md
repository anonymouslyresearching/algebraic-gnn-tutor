# ⚡ Quick Start Guide

Get the Graph Neural Tutor running in under 5 minutes! This guide provides the fastest path from clone to demo.

## 🎯 Choose Your Path

### 🤖 Option 1: Automated Setup (2 minutes)
```bash
# Linux/macOS
chmod +x QUICKSTART.sh && ./QUICKSTART.sh

# Windows
./QUICKSTART.bat
```

### 🛠️ Option 2: Manual Setup (5 minutes)
See [Manual Installation](#manual-installation) below.

---

## 🚀 Automated Setup

### For Linux/macOS Users

```bash
# 1. Download and run setup script
# Clone the repository
git clone https://github.com/anonymouslyresearching/algebraic-gnn-tutor.git
cd algebraic-gnn-tutor
chmod +x QUICKSTART.sh
./QUICKSTART.sh
```

### For Windows Users

```cmd
REM 1. Clone the repository
git clone https://github.com/anonymouslyresearching/algebraic-gnn-tutor.git
cd graph-neural-tutor

REM 2. Run setup script
QUICKSTART.bat
```

### What the Script Does

The automated setup will:
- ✅ Check Python 3.8+ installation
- ✅ Create and activate virtual environment
- ✅ Install PyTorch (CPU version for faster setup)
- ✅ Install all required dependencies
- ✅ Generate sample dataset
- ✅ Test the installation
- ✅ Provide next steps

---

## 🛠️ Manual Installation

### Step 1: Prerequisites

**Python Requirements:**
- Python 3.8 or higher (3.10+ recommended)
- pip package manager
- 4GB+ RAM (8GB+ recommended for model training)

**Check Your Python:**
```bash
python --version  # Should be 3.8+
pip --version     # Should be available
```

### Step 2: Clone Repository

```bash
git clone https://github.com/anonymouslyresearching/algebraic-gnn-tutor.git
cd graph-neural-tutor
```

### Step 3: Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for faster setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### Step 5: Setup Project

```bash
# Create necessary directories
mkdir -p models datasets results figures web/static

# Generate sample dataset
python -c "
from main import create_robust_dataset
import torch
import os

print('Generating sample dataset...')
dataset = create_robust_dataset(num_per_rule=50, num_neg=100)
os.makedirs('datasets', exist_ok=True)
torch.save(dataset, 'datasets/sample_dataset.pt')
print(f'Sample dataset created with {len(dataset)} examples')
"
```

### Step 6: Test Installation

```bash
# Test core functionality
python -c "
import torch
import sympy
from main import DistinctAlgebraicGNN, create_ast_graph
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"

# Test web application
python -c "
from web.app import app
print('✅ Web application loads successfully')
"
```

---

## 🎮 Usage Examples

### Start the Web Interface

```bash
# Start Flask development server
python web/app.py

# Or use the lite version (faster startup)
python web/app_lite.py

# Access at: http://localhost:5000
```

### Start the API Server

```bash
# Start FastAPI server
python -m uvicorn api.main:app --reload

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Run Model Training

```bash
# Quick training (small dataset)
python -c "from main import quick_run; quick_run()"

# Full experimental pipeline
python main.py
```

### Test the System

```bash
# Run comprehensive tests
python test_system.py

# Test mathematical robustness
python test_robust_algebra.py
```

---

## 🌐 API Usage

### Test with curl

```bash
# Health check
curl http://localhost:8000/api/health

# Get examples
curl http://localhost:8000/api/examples

# Analyze transformation
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "original": "3*x + 5 = 11",
    "transformed": "3*x = 6",
    "model": "main"
  }'
```

### Python Client Example

```python
import requests

# Setup
base_url = "http://localhost:8000"

# Analyze equation transformation
response = requests.post(f"{base_url}/api/analyze", json={
    "original": "3*x + 5 = 11",
    "transformed": "3*x = 6",
    "model": "main"
})

result = response.json()
print(f"Rule: {result['result']['rule']}")
print(f"Valid: {result['result']['validity']}")
print(f"Confidence: {result['result']['confidence']:.2f}")
```

---

## 📊 Example Transformations

Try these examples in the web interface:

### Basic Algebra
```
Original: 3*x + 5 = 11
Transformed: 3*x = 6
Expected: sub_const (subtract constant)
```

### Solving for Variable
```
Original: 2*x = 8
Transformed: x = 4
Expected: div_coeff (divide by coefficient)
```

### Factoring
```
Original: x**2 + 5*x + 6 = 0
Transformed: (x+2)*(x+3) = 0
Expected: factor (factorization)
```

### Expanding
```
Original: (x+1)*(x+2) = 0
Transformed: x**2 + 3*x + 2 = 0
Expected: expand (expansion)
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional configuration
export DEBUG=true           # Enable debug mode
export PORT=5000           # Web server port
export MODEL_TYPE=main     # Default model (main/simple/minimal)
export DEVICE=cpu          # Force CPU usage
```

### Model Selection

The system includes three models:

1. **GNT-Main** (default): Full GNN with attention and uncertainty
2. **GNT-Simple**: Simplified GCN for comparison
3. **GNT-Minimal**: MLP baseline

Change in the web interface or via API:
```python
# In web interface: select from dropdown
# In API: specify "model": "simple" in request
```

---

## 🚨 Troubleshooting

### Common Issues

#### Python Version Error
```
Error: Python 3.8+ required
Solution: Update Python or use pyenv/conda
```

#### Import Errors
```
Error: ModuleNotFoundError: No module named 'torch'
Solution: Ensure virtual environment is activated and dependencies installed
```

#### Port Already in Use
```
Error: Port 5000 is already in use
Solution: Kill existing process or use different port:
python web/app.py --port 5001
```

#### Memory Issues
```
Error: CUDA out of memory
Solution: Use CPU version or reduce batch size:
export DEVICE=cpu
```

### Getting Help

1. **Check logs**: Look for error messages in terminal output
2. **Test installation**: Run `python test_system.py`
3. **Verify environment**: Check Python version and virtual environment
4. **Clean install**: Delete `venv/` folder and reinstall
5. **Issue tracker**: Report bugs with full error messages

### Performance Tips

- **Use CPU version** for development (faster installation)
- **Enable GPU** for training large models
- **Use lite version** for quick demos
- **Local deployment** for testing

---

## 📚 Next Steps

### Learn More
- **📖 Full Documentation**: See `docs/` directory
- **🔌 API Reference**: Visit `/docs` endpoint when server is running
- **🧪 Advanced Usage**: Read `CONTRIBUTING.md`
- **🚀 Deployment**: Check `DEPLOYMENT_GUIDE.md`

### Explore Features
- **Interactive Demo**: Web interface with real-time analysis
- **API Integration**: RESTful endpoints for programmatic access
- **Model Training**: Experiment with different architectures
- **Educational Tools**: Classroom-ready demonstrations

### Contribute
- **Report Issues**: Use GitHub issue tracker
- **Suggest Features**: Propose educational enhancements
- **Submit Research**: Academic collaborations welcome
- **Improve Documentation**: Help make setup even easier

---

## ✅ Success Checklist

After setup, you should be able to:

- [ ] **Web Interface**: Access http://localhost:5000 and analyze equations
- [ ] **API Access**: Get response from http://localhost:8000/api/health
- [ ] **Model Training**: Run `python main.py` without errors
- [ ] **Tests Pass**: Execute `python test_system.py` successfully
- [ ] **Examples Work**: Try sample transformations in web interface

If all boxes are checked, you're ready to explore the Graph Neural Tutor! 🎉

---

**🚀 Happy Exploring!**

*This quickstart guide gets you up and running fast. For detailed documentation, deployment options, and advanced features, see the other files in this repository.*