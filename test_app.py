"""
Quick Test Script
Verifies that the application structure is working correctly
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60 + "\n")

    try:
        print("Importing config...", end=" ")
        import config
        print("✓")

        print("Importing data_fetcher...", end=" ")
        from data_fetcher import DataFetcher
        print("✓")

        print("Importing indicators...", end=" ")
        from indicators import TechnicalIndicators
        print("✓")

        print("Importing visualizer...", end=" ")
        from visualizer import StockVisualizer
        print("✓")

        print("Importing analyzer...", end=" ")
        from analyzer import StockAnalyzer
        print("✓")

        print("\nAll imports successful!\n")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}\n")
        return False


def test_configuration():
    """Test configuration"""
    print("Testing Configuration...")
    print("="*60 + "\n")

    try:
        from config import (
            RSI_PERIOD, MACD_FAST, MACD_SLOW,
            RSI_OVERBOUGHT, RSI_OVERSOLD
        )

        print(f"RSI Period: {RSI_PERIOD}")
        print(f"RSI Overbought: {RSI_OVERBOUGHT}")
        print(f"RSI Oversold: {RSI_OVERSOLD}")
        print(f"MACD Fast: {MACD_FAST}")
        print(f"MACD Slow: {MACD_SLOW}")
        print("\nConfiguration loaded successfully!\n")
        return True

    except Exception as e:
        print(f"Configuration test failed: {e}\n")
        return False


def test_analyzer_creation():
    """Test analyzer creation (without fetching data)"""
    print("Testing Analyzer Creation...")
    print("="*60 + "\n")

    try:
        from analyzer import StockAnalyzer

        analyzer = StockAnalyzer("AAPL", period="1mo", interval="1d")
        print(f"✓ StockAnalyzer created for {analyzer.symbol}")
        print(f"  Period: {analyzer.period}")
        print(f"  Interval: {analyzer.interval}\n")

        return True

    except Exception as e:
        print(f"Analyzer creation failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "Stock Analysis Application - Quick Test".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Analyzer Creation", test_analyzer_creation()))

    # Summary
    print("="*60)
    print("Test Summary:")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Application is ready to use.\n")
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run analysis: python main.py")
        print("3. Or run examples: python examples.py\n")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
