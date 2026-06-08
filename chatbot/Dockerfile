FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -U pip && \
    pip3 install --no-cache-dir -r requirements.txt


ENV APP_NAME="Robi AI Chatbot API"
ENV ENV="dev"
ENV SESSION_TTL_SECONDS=900
ENV LOG_LEVEL="INFO"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]