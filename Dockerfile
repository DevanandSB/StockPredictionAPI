# --- Build Stage ---
# This stage installs dependencies and builds the application.
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Final Stage ---
# This stage creates the final, lean production image.
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the pre-built Python wheels from the builder stage
COPY --from=builder /wheels /wheels

# Install the Python dependencies from the local wheels
RUN pip install --no-cache /wheels/*

# Copy the application code into the final image
COPY . .

# Create the data directory (if it's not already there)
RUN mkdir -p app/data

# Expose the port the application will run on
EXPOSE 8000

# Set the command to run the application
# Use gunicorn for a production-ready WSGI server
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]