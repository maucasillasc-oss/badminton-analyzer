FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libxcb1 \
    ffmpeg wget unzip curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download hit-frame detection model weights
# Models from: https://github.com/arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis
# Weights hosted on Google Drive by the paper authors
COPY download_weights.py .
RUN python download_weights.py hitframe_weights

# Copy all application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output

# Set environment variable for weights directory
ENV HITFRAME_WEIGHTS_DIR=/app/hitframe_weights

# Expose port
EXPOSE 8000

# Start command with extended timeout for video processing
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "900", "--workers", "1"]
