FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3 and required system audio/network dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Create cache directories and set permissions
RUN mkdir -p /app/models /app/audio_cache && chmod 777 /app/audio_cache

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -U pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

EXPOSE 8000

CMD ["python3", "app.py"]