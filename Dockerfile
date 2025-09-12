FROM python:3.12-slim

WORKDIR /app

COPY docker_requirements.txt .

# Install all packages at once
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker_requirements.txt

# Install torch with CPU version
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]