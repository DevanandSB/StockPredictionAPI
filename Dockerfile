# Use the full python image to get build tools if needed
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install the TA-Lib C-library dependency first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y build-essential wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# --- GRANULAR BATCH INSTALLATION TO MANAGE MEMORY ---

# [cite_start]Batch 1: The heaviest ML libraries (PyTorch, Transformers) [cite: 1]
RUN pip install --no-cache-dir \
    torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers==4.56.1 \
    tokenizers==0.22.0 \
    safetensors==0.6.2

# [cite_start]Batch 2: The core data science stack [cite: 1]
RUN pip install --no-cache-dir \
    pandas==2.3.0 \
    numpy==2.3.1 \
    scikit-learn==1.7.1 \
    scipy==1.16.1 \
    plotly==6.3.0 \
    statsmodels==0.14.5

# [cite_start]Batch 3: Web framework and asynchronous libraries [cite: 1]
RUN pip install --no-cache-dir \
    fastapi==0.103.2 \
    uvicorn==0.23.2 \
    aiohttp==3.12.15 \
    requests==2.31.0 \
    Jinja2==3.1.4 \
    gunicorn

# [cite_start]Batch 4: Finance, NLP, and parsing libraries [cite: 1]
RUN pip install --no-cache-dir \
    yfinance==0.2.65 \
    nsepy==0.8 \
    TA-Lib==0.4.28 \
    nltk==3.9.1 \
    vaderSentiment==3.3.2 \
    beautifulsoup4==4.12.2 \
    lxml==5.3.1

# Batch 5: The remaining packages. This final step is now much smaller and safer.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Command to run your application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]