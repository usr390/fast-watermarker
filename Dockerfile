# Dockerfile
FROM python:3.11-slim

# 1) OS deps: ffmpeg + a readable font
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# 2) App directory
WORKDIR /app

# 3) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy source
COPY . .

# 5) Font env (optional if you vendor your own TTF in the repo)
ENV WATERMARK_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

# 6) Start command (Railway sets $PORT)
CMD sh -c 'hypercorn main:app --bind 0.0.0.0:${PORT:-8000}'