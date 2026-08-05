FROM python:3.11-slim-bookworm

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libxcb1 \
    ffmpeg wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de la app
WORKDIR /app

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar pesos de TrackNetV3
RUN mkdir -p ckpts && \
    python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA', '/tmp/ckpts.zip', quiet=False)" && \
    cd /tmp && unzip ckpts.zip && \
    find /tmp -name 'TrackNet_best.pt' -exec cp {} /app/ckpts/ \; && \
    find /tmp -name 'InpaintNet_best.pt' -exec cp {} /app/ckpts/ \; && \
    rm -rf /tmp/ckpts.zip /tmp/TrackNetV3* && \
    ls -la /app/ckpts/

# Pre-descargar modelo YOLO
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copiar el resto de la app
COPY . .

# Crear directorios necesarios
RUN mkdir -p uploads output

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1"]
