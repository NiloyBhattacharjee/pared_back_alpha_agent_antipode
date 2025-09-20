FROM python:3.10-slim

# Environment hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# System deps for scientific stack and matplotlib runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ gfortran \
    libopenblas-dev liblapack-dev \
    libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
# Ensure we have the latest pip/setuptools/wheel to prefer prebuilt wheels
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src ./src
COPY run.py ./run.py
COPY data ./data
COPY outputs ./outputs
COPY tests ./tests

# Default command (configurable at runtime)
CMD ["python", "run.py"]
