#!/bin/bash

source .venv/bin/activate

echo "Checking installed packages..."
pip list | grep -E "fastapi|uvicorn|pydantic|torch|scikit-learn|pandas|numpy"

echo ""
echo "Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Testing FastAPI installation..."
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"