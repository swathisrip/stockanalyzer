#!/bin/bash
# Quick Start Script for Stock Analysis Application

echo "========================================"
echo "Stock Technical Analysis Setup"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✓ Python is installed"
python --version
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import yfinance, pandas, matplotlib" 2>/dev/null && echo "✓ All dependencies installed successfully" || echo "❌ Failed to install dependencies"
echo ""

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  1. Basic usage:     python main.py"
echo "  2. Interactive:     python main.py (then uncomment interactive_mode)"
echo "  3. Advanced examples: python examples.py"
echo ""
echo "For more information, see README.md"
