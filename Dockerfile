# Stage 1: Builder
FROM python:3.11 as builder

WORKDIR /app

# Download pre-built TA-Lib wheel
RUN wget https://github.com/mrjbq7/ta-lib/releases/download/TA_Lib-0.6.7/TA_Lib-0.6.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Stage 2: Final image
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built TA-Lib wheel from builder
COPY --from=builder /app/TA_Lib-0.6.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .

# Install TA-Lib from pre-built wheel
RUN pip install --no-cache-dir TA_Lib-0.6.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY requirements.txt .

# Install packages in batches
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy requests scikit-learn Jinja2 gunicorn
RUN pip install --no-cache-dir yfinance newsapi-python nltk python-multipart beautifulsoup4 nsepy
RUN pip install --no-cache-dir vaderSentiment googlesearch-python scipy arch plotly google-cloud-storage
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]