"""
Advanced Usage Examples
Demonstrates advanced features of the stock analysis application
"""

from analyzer import StockAnalyzer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_single_stock_detailed_analysis():
    """Example 1: Single stock with detailed analysis"""
    print("\n" + "="*60)
    print("Example 1: Detailed Analysis of Tesla")
    print("="*60 + "\n")

    analyzer = StockAnalyzer("TSLA", period="6mo", interval="1d")

    if analyzer.analyze():
        # Print summary
        analyzer.print_analysis_summary()

        # Get detailed analysis
        analysis = analyzer.get_latest_analysis()
        print("\nDetailed Metrics:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        # Get raw data for custom analysis
        data = analyzer.data
        print(f"\nData shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")

        # Custom analysis: Calculate win rate
        signals = analyzer.signals
        buy_signals = len(signals[signals['Buy_Signal'] == 1])
        sell_signals = len(signals[signals['Sell_Signal'] == 1])

        print(f"\nSignal Summary (6 months):")
        print(f"  Buy Signals: {buy_signals}")
        print(f"  Sell Signals: {sell_signals}")

        # Save analysis
        analyzer.export_analysis_to_csv("TSLA_detailed_analysis.csv")
        print("\n✓ Analysis exported to TSLA_detailed_analysis.csv")


def example_2_compare_multiple_stocks():
    """Example 2: Compare multiple tech stocks"""
    print("\n" + "="*60)
    print("Example 2: Comparing Tech Stocks")
    print("="*60 + "\n")

    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    results = []

    for symbol in tech_stocks:
        try:
            analyzer = StockAnalyzer(symbol, period="3mo")
            if analyzer.analyze():
                analysis = analyzer.get_latest_analysis()
                results.append({
                    'Symbol': symbol,
                    'Price': analysis['Close Price'],
                    'RSI': analysis['RSI'],
                    'MACD': analysis['MACD'],
                    'Trend': 'BULLISH' if analysis['MACD'] > analysis['MACD Signal'] else 'BEARISH'
                })
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    # Identify strongest performers
    print("\n\nAnalysis:")
    print(f"Strongest RSI Signal: {comparison_df.loc[comparison_df['RSI'].idxmin(), 'Symbol']} (RSI: {comparison_df['RSI'].min():.2f})")
    print(f"Bullish Stocks: {len(comparison_df[comparison_df['Trend'] == 'BULLISH'])}")
    print(f"Bearish Stocks: {len(comparison_df[comparison_df['Trend'] == 'BEARISH'])}")


def example_3_rsi_extremes():
    """Example 3: Find RSI extremes"""
    print("\n" + "="*60)
    print("Example 3: RSI Extreme Analysis")
    print("="*60 + "\n")

    analyzer = StockAnalyzer("AAPL", period="1y", interval="1d")

    if analyzer.analyze():
        data = analyzer.data

        # Find extremes
        max_rsi = data['RSI'].max()
        min_rsi = data['RSI'].min()
        avg_rsi = data['RSI'].mean()

        print(f"RSI Statistics (1 Year):")
        print(f"  Max RSI: {max_rsi:.2f}")
        print(f"  Min RSI: {min_rsi:.2f}")
        print(f"  Average RSI: {avg_rsi:.2f}")
        print(f"  Current RSI: {data['RSI'].iloc[-1]:.2f}")

        # Count overbought/oversold periods
        overbought = len(data[data['RSI'] > 70])
        oversold = len(data[data['RSI'] < 30])
        total_days = len(data)

        print(f"\nRSI Distribution:")
        print(f"  Overbought Days (>70): {overbought} ({overbought/total_days*100:.1f}%)")
        print(f"  Oversold Days (<30): {oversold} ({oversold/total_days*100:.1f}%)")
        print(f"  Normal Range (30-70): {total_days - overbought - oversold} ({(total_days - overbought - oversold)/total_days*100:.1f}%)")


def example_4_macd_crossovers():
    """Example 4: Identify MACD crossovers"""
    print("\n" + "="*60)
    print("Example 4: MACD Crossover Analysis")
    print("="*60 + "\n")

    analyzer = StockAnalyzer("MSFT", period="3mo", interval="1d")

    if analyzer.analyze():
        data = analyzer.data

        # Find crossovers
        crossovers = []
        for i in range(1, len(data)):
            prev_diff = data['MACD'].iloc[i-1] - data['MACD_Signal'].iloc[i-1]
            curr_diff = data['MACD'].iloc[i] - data['MACD_Signal'].iloc[i]

            # Bullish crossover (MACD crosses above signal)
            if prev_diff < 0 and curr_diff > 0:
                crossovers.append({
                    'Date': data.index[i],
                    'Type': 'BULLISH',
                    'Price': data['Close'].iloc[i],
                    'MACD': data['MACD'].iloc[i],
                    'Signal': data['MACD_Signal'].iloc[i]
                })

            # Bearish crossover (MACD crosses below signal)
            elif prev_diff > 0 and curr_diff < 0:
                crossovers.append({
                    'Date': data.index[i],
                    'Type': 'BEARISH',
                    'Price': data['Close'].iloc[i],
                    'MACD': data['MACD'].iloc[i],
                    'Signal': data['MACD_Signal'].iloc[i]
                })

        if crossovers:
            crossover_df = pd.DataFrame(crossovers)
            print("MACD Crossovers (Last 3 Months):")
            print(crossover_df.to_string(index=False))

            # Summary
            bullish = len(crossover_df[crossover_df['Type'] == 'BULLISH'])
            bearish = len(crossover_df[crossover_df['Type'] == 'BEARISH'])
            print(f"\nSummary:")
            print(f"  Bullish Crossovers: {bullish}")
            print(f"  Bearish Crossovers: {bearish}")
        else:
            print("No MACD crossovers found in this period")


def example_5_moving_average_analysis():
    """Example 5: Moving average trend analysis"""
    print("\n" + "="*60)
    print("Example 5: Moving Average Trend Analysis")
    print("="*60 + "\n")

    analyzer = StockAnalyzer("GOOGL", period="1y", interval="1d")

    if analyzer.analyze():
        data = analyzer.data

        # Get latest values
        latest_price = data['Close'].iloc[-1]
        ema_20 = data['EMA_20'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        ema_200 = data['EMA_200'].iloc[-1]

        print(f"Current Price: ${latest_price:.2f}")
        print(f"\nMoving Averages:")
        print(f"  EMA 20: ${ema_20:.2f}")
        print(f"  EMA 50: ${ema_50:.2f}")
        print(f"  EMA 200: ${ema_200:.2f}")

        # Trend analysis
        print(f"\nTrend Analysis:")
        if latest_price > ema_20 > ema_50 > ema_200:
            print("  ✓ Strong Uptrend (Price > EMA20 > EMA50 > EMA200)")
        elif latest_price > ema_50 > ema_200:
            print("  ✓ Moderate Uptrend")
        elif latest_price < ema_20 < ema_50 < ema_200:
            print("  ✗ Strong Downtrend (Price < EMA20 < EMA50 < EMA200)")
        elif latest_price < ema_50 < ema_200:
            print("  ✗ Moderate Downtrend")
        else:
            print("  → Consolidation/Undefined Trend")

        # Golden/Death Cross detection
        if ema_50 > ema_200:
            print("  ★ Golden Cross detected (EMA50 > EMA200) - Long-term bullish")
        elif ema_50 < ema_200:
            print("  ★ Death Cross detected (EMA50 < EMA200) - Long-term bearish")


def example_6_volatility_analysis():
    """Example 6: Volatility analysis with Bollinger Bands"""
    print("\n" + "="*60)
    print("Example 6: Volatility Analysis")
    print("="*60 + "\n")

    analyzer = StockAnalyzer("AMZN", period="6mo", interval="1d")

    if analyzer.analyze():
        data = analyzer.data

        # Calculate price range
        price_range = data['Close'].max() - data['Close'].min()
        avg_price = data['Close'].mean()
        volatility = (price_range / avg_price) * 100

        # Bollinger Band analysis
        latest_price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_width = bb_upper - bb_lower

        print(f"Current Price: ${latest_price:.2f}")
        print(f"\nVolatility Metrics:")
        print(f"  Price Range (6mo): ${price_range:.2f}")
        print(f"  Volatility: {volatility:.2f}%")
        print(f"  Average Price: ${avg_price:.2f}")

        print(f"\nBollinger Bands:")
        print(f"  Upper Band: ${bb_upper:.2f}")
        print(f"  Middle Band: ${bb_middle:.2f}")
        print(f"  Lower Band: ${bb_lower:.2f}")
        print(f"  Band Width: ${bb_width:.2f}")

        # Position within bands
        if latest_price > bb_upper:
            print(f"  Price is ABOVE upper band (potential resistance)")
        elif latest_price < bb_lower:
            print(f"  Price is BELOW lower band (potential support)")
        else:
            position = ((latest_price - bb_lower) / (bb_upper - bb_lower)) * 100
            print(f"  Price is within bands ({position:.1f}% of band width)")


def run_all_examples():
    """Run all examples"""
    try:
        example_1_single_stock_detailed_analysis()
        example_2_compare_multiple_stocks()
        example_3_rsi_extremes()
        example_4_macd_crossovers()
        example_5_moving_average_analysis()
        example_6_volatility_analysis()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_single_stock_detailed_analysis()
    # example_2_compare_multiple_stocks()
    # example_3_rsi_extremes()
    # example_4_macd_crossovers()
    # example_5_moving_average_analysis()
    # example_6_volatility_analysis()
