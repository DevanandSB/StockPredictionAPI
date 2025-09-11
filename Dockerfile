FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install packages in batches to avoid memory issues
RUN pip install --no-cache-dir --upgrade pip

# Install pandas-ta dependencies first
RUN pip install --no-cache-dir numba tqdm pandas-ta==0.4.67b0

# Install other packages in batches
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy requests scikit-learn Jinja2 gunicorn
RUN pip install --no-cache-dir yfinance newsapi-python nltk python-multipart beautifulsoup4 nsepy
RUN pip install --no-cache-dir vaderSentiment googlesearch-python scipy arch plotly google-cloud-storage

# Install torch with CPU version
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir transformers==4.56.1 lightning-fabric==2.5.5

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]