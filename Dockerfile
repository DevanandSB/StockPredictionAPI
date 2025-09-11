FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install with older pip version that handles index-url correctly
RUN pip install --no-cache-dir --upgrade "pip==23.3.2" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]