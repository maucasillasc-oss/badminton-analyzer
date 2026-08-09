FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output hitframe_weights

# Download trained model weights from S3 at build time
# OPT Transformer (793MB) + scaler + SA-CNN
RUN python download_weights.py

# Set environment variable for weights directory
ENV HITFRAME_WEIGHTS_DIR=/app/hitframe_weights

# Expose port
EXPOSE 8000

# Start command with extended timeout for video processing
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "900", "--workers", "1"]
