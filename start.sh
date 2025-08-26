#!/bin/bash

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
pip install -r requirements.txt

# Check if PyTorch installed correctly, if not install with CPU support
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "PyTorch not installed correctly, installing CPU version..."
    pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
}

# Create data directory if it doesn't exist
mkdir -p app/data

echo "Setup complete. Please make sure your model files are in app/data/:"
echo "- model_weights.pt"
echo "- features.pkl"
echo "- target.pkl"
echo "- metadata.pkl"

echo "To start the server:"
echo "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"