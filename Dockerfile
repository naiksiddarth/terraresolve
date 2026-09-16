FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch install is separate and explicit — swap this line if your HF Space has GPU
# GPU: --index-url https://download.pytorch.org/whl/cu128
# CPU (default, safe assumption for HF Spaces free tier):
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data

COPY . .
# models/pretrained/ and demo/scenes/ must already contain real weights + precomputed
# assets BEFORE this build runs — they get baked into the image here, deliberately,
# so the demo never depends on a live download during judging.

EXPOSE 7860
ENV WEIGHTS_DIR=models/pretrained/sen2sr_rgbn_x4
ENV DATA_DIR=data
ENV DB_PATH=/app/data/terraresolve.db

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
