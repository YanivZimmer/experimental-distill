#!/bin/bash
# Simple setup script for UV environment

set -e  # Exit on error

echo "================================"
echo "Distillation Training Setup"
echo "================================"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ UV is installed ($(uv --version))"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Initialize UV environment
echo ""
echo "Setting up UV environment..."
uv sync

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands directly with UV:"
echo "  uv run python training/local_entry.py"
echo ""
echo "Or use Make:"
echo "  make train-local"
echo ""
