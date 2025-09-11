FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install torch first with the specific index URL (smaller version)
RUN pip install --no-cache-dir torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install packages in batches to reduce memory usage
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy requests scikit-learn Jinja2 gunicorn
RUN pip install --no-cache-dir yfinance newsapi-python nltk python-multipart beautifulsoup4 nsepy
RUN pip install --no-cache-dir vaderSentiment googlesearch-python scipy arch plotly google-cloud-storage ta

COPY . .

EXPOSE 8000

# Use fewer workers to reduce memory usage
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]