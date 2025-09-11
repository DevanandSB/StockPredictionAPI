FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install torch first with the specific index URL
RUN pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Now install all other packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]