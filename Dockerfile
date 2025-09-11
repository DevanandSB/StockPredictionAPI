FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install torch first with the specific index URL
RUN pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Now install all other packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    yfinance \
    requests \
    newsapi-python \
    transformers \
    nltk \
    scikit-learn \
    Jinja2 \
    python-multipart \
    beautifulsoup4 \
    nsepy \
    vaderSentiment \
    googlesearch-python \
    lightning-fabric \
    scipy \
    arch \
    plotly \
    gunicorn \
    google-cloud-storage \
    pandas-ta  # Using pandas-ta instead

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]