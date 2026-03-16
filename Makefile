.PHONY: help install install-mac install-cloud clean test lint format train-local train-cloud prepare split docker-build docker-clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install base dependencies with uv"
	@echo "  make install-mac     - Install Mac-specific dependencies"
	@echo "  make install-cloud   - Install cloud/GPU dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo ""
	@echo "  make prepare         - Prepare dataset (filter by Gemini agreement)"
	@echo "  make split           - Split dataset into train/val/test"
	@echo "  make train-local     - Train locally (0.5B model, 512 tokens)"
	@echo "  make train-cloud     - Train on cloud (3B model, 32K tokens)"
	@echo "  make train-mock      - Mock training (testing, no actual training)"
	@echo ""
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linting (ruff)"
	@echo "  make format          - Format code (black)"
	@echo "  make clean           - Clean build artifacts"
	@echo ""
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-clean    - Clean Docker images and containers"
	@echo ""
	@echo "  make all             - Prepare, split, and train locally"


setup:
	uv sync

# Installation targets
install:
	@echo "Installing base dependencies with uv..."
	uv pip install -e .

install-mac:
	@echo "Installing Mac-specific dependencies..."
	uv pip install -e ".[mac]"

install-cloud:
	@echo "Installing cloud/GPU dependencies..."
	uv pip install -e ".[cloud]"

install-dev:
	@echo "Installing development dependencies..."
	uv pip install -e ".[dev]"

# Data preparation
prepare:
	@echo "Preparing dataset (filtering by Gemini agreement)..."
	python scripts/prepare_dataset.py

split:
	@echo "Splitting dataset into train/val/test..."
	python scripts/split_dataset.py

# Training targets
train-local:
	@echo "Starting local training (Mac)..."
	uv run python training/local_entry.py

train-cloud:
	@echo "Starting cloud training..."
	python training/cloud_entry.py

train-mock:
	@echo "Starting mock training (testing)..."
	python training/mock_entry.py

# Testing and quality
test:
	@echo "Running tests..."
	python scripts/test_dataset.py --test-splits
	python scripts/test_model.py

lint:
	@echo "Running linting..."
	ruff check .

format:
	@echo "Formatting code..."
	black .

# Docker targets
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-build-mac:
	@echo "Building Mac Docker image..."
	docker-compose build train-mac

docker-build-cloud:
	@echo "Building cloud Docker image..."
	docker-compose build train-cloud

docker-train-mac:
	@echo "Running training in Docker (Mac)..."
	docker-compose run --rm train-mac

docker-train-cloud:
	@echo "Running training in Docker (cloud)..."
	docker-compose run --rm train-cloud

docker-clean:
	@echo "Cleaning Docker images and containers..."
	docker-compose down
	docker rmi simple-distill:mac-train simple-distill:cloud-train 2>/dev/null || true

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build dist

# Complete pipeline
all: prepare split train-local

# Development workflow
dev-setup: install-dev install-mac
	@echo "Development environment ready!"

# Quick validation
validate: lint test
	@echo "Validation complete!"

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
