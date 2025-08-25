@echo off
REM 🚀 Graph Neural Tutor - Quick Start Script
REM Automated setup for Windows systems

echo 🎯 Graph Neural Tutor - Quick Start Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Found Python %PYTHON_VERSION%

echo 📦 Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️  Virtual environment already exists
)

echo 🔄 Activating virtual environment...
call venv\Scripts\activate

echo 📥 Installing dependencies...
echo    Upgrading pip...
python -m pip install --upgrade pip

echo    Installing PyTorch (CPU version for faster setup)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo    Installing other dependencies...
python -m pip install -r requirements.txt

echo 📁 Setting up directories...
if not exist models mkdir models
if not exist datasets mkdir datasets
if not exist results mkdir results
if not exist figures mkdir figures
if not exist web\static mkdir web\static
if not exist web\templates mkdir web\templates

REM Create .gitkeep files
echo. > models\.gitkeep
echo. > datasets\.gitkeep
echo. > results\.gitkeep
echo. > figures\.gitkeep

echo 🔬 Generating sample dataset...
python -c "from main import create_robust_dataset; import torch; import os; print('Generating sample dataset...'); dataset = create_robust_dataset(num_per_rule=50, num_neg=100); os.makedirs('datasets', exist_ok=True); torch.save(dataset, 'datasets/sample_dataset.pt'); print(f'Sample dataset saved with {len(dataset)} examples')"

echo 🧪 Testing installation...
python -c "import torch; import sympy; import flask; from main import create_ast_graph, DistinctAlgebraicGNN; print('✅ All imports successful'); print(f'PyTorch version: {torch.__version__}'); print(f'Device available: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo Next steps:
echo 1. Start the web interface:
echo    python web/app.py
echo.
echo 2. Or run the training pipeline:
echo    python main.py
echo.
echo 3. Or start the API server:
echo    python -m uvicorn api.main:app --reload
echo.
echo 🌐 Web interface will be available at: http://localhost:5000
echo 🔌 API documentation will be at: http://localhost:8000/docs
echo.
echo 📚 For more information, see:
echo    - README.md for full documentation
echo    - QUICKSTART.md for detailed instructions
echo    - docs/ directory for advanced topics
echo.
echo 🚀 Happy researching!
echo.
pause
