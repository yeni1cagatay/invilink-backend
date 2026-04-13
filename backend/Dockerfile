FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# TensorFlow 2.x (TF1 modellerini compat.v1 moduyla yukler)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "tensorflow-cpu==2.9.3"

# Uygulama bagimliliklari
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/model_cache

# v4
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
