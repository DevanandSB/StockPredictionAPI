FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# First install torch with the specific index
RUN pip install --no-cache-dir torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install all other packages from PyPI
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy yfinance requests \
    newsapi-python transformers nltk scikit-learn Jinja2 python-multipart \
    beautifulsoup4 nsepy vaderSentiment googlesearch-python lightning-fabric \
    talib-binary scipy arch plotly gunicorn google-cloud-storage

# Copy the rest of your application's code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 8000

# Set the default command to run the application using a production server
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]