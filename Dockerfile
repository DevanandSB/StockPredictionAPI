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

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]