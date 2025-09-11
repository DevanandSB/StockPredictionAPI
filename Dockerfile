FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install TA-Lib first using pip (it will find the appropriate wheel)
RUN pip install --no-cache-dir TA-Lib==0.6.7

# Install torch with CPU version
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install packages in batches to reduce memory usage
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy requests scikit-learn Jinja2 gunicorn
RUN pip install --no-cache-dir yfinance newsapi-python nltk python-multipart beautifulsoup4 nsepy
RUN pip install --no-cache-dir vaderSentiment googlesearch-python scipy arch plotly google-cloud-storage

# Install remaining packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]