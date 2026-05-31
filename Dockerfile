# ==========================================
# Production-Grade Dockerfile
# Multi-stage/Slim configuration for FastAPI
# ==========================================

FROM python:3.11-slim as base

# Set environment variables for Python execution optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for health check, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project source code
COPY . .

# Create a non-privileged user for process isolation security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

# Expose API runtime port
EXPOSE 8000

# Define container health check using curl hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
