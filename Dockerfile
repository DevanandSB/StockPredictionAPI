# Stage 1: Final image
# We use the full python image to get build tools if needed
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install the TA-Lib C-library dependency first
# This is the most reliable method for installing TA-Lib
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

# --- BATCH INSTALLATION TO MANAGE MEMORY ---

# Batch 1: Install the heaviest ML libraries first (PyTorch, Transformers)
RUN pip install --no-cache-dir \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.56.1 \
    tokenizers==0.22.0 \
    safetensors==0.6.2

# Batch 2: Install the core data science stack
RUN pip install --no-cache-dir \
    pandas==2.3.0 \
    numpy==2.3.1 \
    scikit-learn==1.7.1 \
    scipy==1.16.1 \
    plotly==6.3.0

# Batch 3: Install the remaining packages
# This will be much lighter as the biggest dependencies are already installed.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Command to run your application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]