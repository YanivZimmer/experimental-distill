# Multi-stage build for efficient cloud training
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY scripts/ ./scripts/
COPY training/ ./training/
COPY prompts/ ./prompts/

# Set entrypoint
ENV PYTHONUNBUFFERED=1
CMD ["python3", "training/cloud_train.py"]
