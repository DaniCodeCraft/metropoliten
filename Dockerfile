FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry: no virtualenv in container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --only main

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "-m", "vehicle_ocr.cli", "--input", "/app/data/input", "--output", "/app/data/output/results.json"]
