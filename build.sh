#!/bin/bash

echo "Setting up Stock Prediction API..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install --upgrade pip

# Install base requirements
pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 scikit-learn==1.4.0 pandas==2.1.4 numpy==1.26.4 python-multipart==0.0.6 joblib==1.3.2

# Install PyTorch based on system
echo "Detecting system and installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing PyTorch for macOS..."
    pip install torch torchvision torchaudio
else
    # Linux or other systems
    echo "Installing PyTorch CPU version..."
    pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# Create data directory if it doesn't exist
mkdir -p app/data

echo "Setup complete! Please make sure your model files are in app/data/:"
echo "- model_weights.pt"
echo "- features.pkl"
echo "- target.pkl"
echo "- metadata.pkl"
echo ""
echo "To start the server:"
echo "source .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"