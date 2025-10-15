FROM python:3.11-slim

WORKDIR /app

COPY docker_requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install packages in batches
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy requests scikit-learn Jinja2 gunicorn
RUN pip install --no-cache-dir yfinance newsapi-python nltk python-multipart beautifulsoup4 nsepy
RUN pip install --no-cache-dir vaderSentiment googlesearch-python scipy arch plotly google-cloud-storage
RUN pip install --no-cache-dir transformers lightning-fabric

COPY sw.js .
COPY . .

# Use PORT environment variable (Digital Ocean uses 8080)
ENV PORT=8080
EXPOSE 8080

# Use fewer workers to save memory
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:8080", "app.main:app"]