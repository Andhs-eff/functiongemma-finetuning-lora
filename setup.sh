#!/bin/bash
# Setup script for FunctionGemma training with virtual environment
# This keeps everything isolated and doesn't pollute your global Python environment

set -e  # Exit on error

echo "🎵 Setting up FunctionGemma Music Training Environment"
echo "======================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created: venv/"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "======================================================"
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To deactivate when you're done:"
echo "   deactivate"
echo ""
echo "Next steps:"
echo "   1. python scripts/generate_dataset.py"
echo "   2. python scripts/train.py"
echo "   3. python scripts/quick_test.py --model models/music-assistant-*/final"
echo "======================================================"
