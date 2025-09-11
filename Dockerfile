FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# First install core packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.103.2 \
    uvicorn==0.23.2 \
    pandas==2.3.0 \
    numpy==2.3.1 \
    requests==2.31.0 \
    scikit-learn==1.7.1 \
    Jinja2==3.1.4 \
    gunicorn \
    yfinance==0.2.65 \
    nltk==3.9.1 \
    python-multipart==0.0.20 \
    beautifulsoup4==4.12.2 \
    vaderSentiment==3.3.2 \
    googlesearch-python==1.3.0 \
    scipy==1.16.1 \
    plotly==6.3.0 \
    google-cloud-storage

# Install torch with CPU version
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install pandas-ta with its dependencies
RUN pip install --no-cache-dir numba tqdm pandas-ta==0.4.67b0

# Install remaining packages
RUN pip install --no-cache-dir \
    newsapi-python==0.2.7 \
    nsepy==0.8 \
    arch==7.2.0 \
    transformers==4.56.1 \
    lightning-fabric==2.5.5

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]