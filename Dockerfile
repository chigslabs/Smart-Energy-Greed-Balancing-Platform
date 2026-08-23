"""
Dockerfile – Containerized Smart Energy Management Platform
Build:  docker build -t smart-energy-platform .
Run:    docker run -p 8000:8000 --env-file .env smart-energy-platform
"""

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default: run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
