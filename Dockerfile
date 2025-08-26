FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If PyTorch installation fails in requirements, install it separately
RUN python -c "import torch" 2>/dev/null || pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p app/data

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]