FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/models /app/audio_cache && chmod 777 /app/audio_cache

COPY requirements.txt .

RUN pip3 install --no-cache-dir -U pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python3", "app.py"]