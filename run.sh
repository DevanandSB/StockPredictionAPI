#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    # shellcheck disable=SC2046
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Start the FastAPI server
clear
echo "Starting Stock Prediction API..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000