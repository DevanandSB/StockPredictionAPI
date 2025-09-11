FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib* ta-lib-0.4.0-src.tar.gz

COPY requirements.txt .

# Install all packages from requirements.txt (including exact versions)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use fewer workers to reduce memory usage
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]