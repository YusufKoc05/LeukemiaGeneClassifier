# OncoVision AI — TÜBİTAK 2209-A Docker Konteynerizasyonu
# Resmi Python 3.12 Slim Tabanlı İmaj

FROM python:3.12-slim

# Sistem bağımlılıklarını kur (OpenCV / font / derleme kütüphaneleri)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur
WORKDIR /app

# Bağımlılıkları kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını ve modelleri kopyala
COPY . .

# Gradio için portu aç
EXPOSE 7860

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Uygulamayı başlat
CMD ["python", "app.py"]
