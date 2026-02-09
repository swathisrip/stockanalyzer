"""
Stock Analysis Application
Main entry point with examples
"""

import sys
from analyzer import StockAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_stock(symbol: str, period: str = "1y", interval: str = "1d",
                 show_plots: bool = True) -> StockAnalyzer:
    """
    Analyze a stock with technical indicators

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Data period (default '1y')
        interval: Data interval (default '1d')
        show_plots: Whether to display plots

    Returns:
        StockAnalyzer object
    """
    # Create analyzer
    analyzer = StockAnalyzer(symbol, period, interval)

    # Run analysis
    if not analyzer.analyze():
        logger.error(f"Failed to analyze {symbol}")
        return None

    # Print summary
    analyzer.print_analysis_summary()

    # Get recent signals
    recent_signals = analyzer.get_recent_signals(days=20)
    if recent_signals is not None and len(recent_signals) > 0:
        print(f"\nRecent Trading Signals (Last 20 days):")
        print(f"{'Date':<20} {'Close Price':<15} {'Signal':<20} {'Type':<40}")
        print("-" * 95)
        for _, row in recent_signals.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            print(f"{date_str:<20} ${row['Close']:<14.2f} {row['Signal']:<20} {row['Type']:<40}")
    else:
        print(f"\nNo trading signals detected in the last 20 days")
        print(f"Current Market Conditions:")
        if analyzer.data is not None and len(analyzer.data) > 0:
            latest = analyzer.data.iloc[-1]
            print(f"  Latest RSI: {latest['RSI']:.2f}")
            print(f"  Latest MACD: {latest['MACD']:.4f}")
            print(f"  Latest MACD Signal: {latest['MACD_Signal']:.4f}")
            print(f"  (Buy: RSI<30+MACD>Signal, Sell: RSI>70+MACD<Signal, Bullish: MACD>Signal+RSI>40, Bearish: MACD<Signal+RSI<60)")

    # Display plots if requested
    if show_plots:
        print("\nGenerating charts...")
        try:
            analyzer.plot_all_indicators()
        except Exception as e:
            logger.error(f"Error plotting indicators: {e}")

    return analyzer


def compare_stocks(symbols: list, period: str = "1y") -> None:
    """
    Compare multiple stocks

    Args:
        symbols: List of stock symbols
        period: Data period
    """
    analyzers = {}

    print(f"\n{'='*80}")
    print(f"{'Comparing Stocks: ' + ', '.join(symbols):^80}")
    print(f"{'='*80}\n")

    for symbol in symbols:
        try:
            analyzer = StockAnalyzer(symbol, period)
            if analyzer.analyze():
                analyzers[symbol] = analyzer.get_latest_analysis()
            else:
                logger.warning(f"Failed to analyze {symbol}")
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    # Create comparison table
    if analyzers:
        print(f"{'Symbol':<10} {'Price':<12} {'RSI':<8} {'MACD':<12} {'Signal':<12}")
        print("-" * 54)
        for symbol, analysis in analyzers.items():
            if analysis:
                rsi = f"{analysis['RSI']:.2f}" if analysis['RSI'] else "N/A"
                macd = f"{analysis['MACD']:.4f}" if analysis['MACD'] else "N/A"
                signal = f"{analysis['MACD Signal']:.4f}" if analysis['MACD Signal'] else "N/A"
                print(f"{symbol:<10} ${analysis['Close Price']:<11.2f} {rsi:<8} {macd:<12} {signal:<12}")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("Stock Technical Analysis Application")
    print("="*60 + "\n")

    # Example 1: Analyze a single stock
    print("Example 1: Analyzing Apple (AAPL)")
    print("-" * 60)
    analyzer = analyze_stock("AAPL", period="6mo", show_plots=False)

    if analyzer:
        # Export data
        analyzer.export_analysis_to_csv("AAPL_analysis.csv")
        print("\n✓ AAPL analysis complete!")

    # Example 2: Compare multiple stocks
    print("\n\nExample 2: Comparing Multiple Stocks")
    print("-" * 60)
    compare_stocks(["AAPL", "MSFT", "GOOGL"], period="3mo")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60 + "\n")


def interactive_mode():
    """Interactive mode for analyzing custom stocks"""
    print("\n" + "="*60)
    print("Stock Analysis - Interactive Mode")
    print("="*60)

    while True:
        print("\nOptions:")
        print("1. Analyze a stock")
        print("2. Compare multiple stocks")
        print("3. Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
            period = input("Enter period (1d/5d/1mo/3mo/6mo/1y/2y/5y/10y) [default: 1y]: ").strip() or "1y"

            print(f"\nAnalyzing {symbol}...")
            analyzer = analyze_stock(symbol, period=period, show_plots=True)

            if analyzer:
                export = input("\nExport analysis to CSV? (y/n): ").strip().lower()
                if export == 'y':
                    filename = f"{symbol}_analysis.csv"
                    analyzer.export_analysis_to_csv(filename)
                    print(f"✓ Exported to {filename}")

        elif choice == "2":
            symbols_input = input("Enter stock symbols separated by comma (e.g., AAPL,MSFT,GOOGL): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(",")]
            period = input("Enter period (1d/5d/1mo/3mo/6mo/1y/2y/5y/10y) [default: 1y]: ").strip() or "1y"

            compare_stocks(symbols, period=period)

        elif choice == "3":
            print("\nGoodbye!")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    # Run main analysis
    main()

    # Uncomment the line below to run in interactive mode instead
    # interactive_mode()
