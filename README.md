# Stock Technical Analysis Application

A comprehensive Python application for analyzing stocks using technical indicators like MACD (Moving Average Convergence Divergence) and RSI (Relative Strength Index).

## Features

### Technical Indicators
- **RSI (Relative Strength Index)**: Measures momentum and identify overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Identifies trend changes and momentum
- **Exponential Moving Averages (EMA)**: 20, 50, and 200-day EMAs for trend analysis
- **Simple Moving Averages (SMA)**: 20, 50, and 200-day SMAs for trend confirmation
- **Bollinger Bands**: Identify volatility and support/resistance levels

### Trading Signals
- **Buy Signals**: Generated when RSI is oversold AND MACD shows bullish crossover
- **Sell Signals**: Generated when RSI is overbought AND MACD shows bearish crossover
- **Signal Analysis**: Recent signals over customizable time periods

### Visualization
- Price charts with moving averages
- RSI indicator charts with overbought/oversold zones
- MACD charts with signal line and histogram
- Bollinger Bands visualization
- Comprehensive multi-indicator dashboard
- Volume analysis

### Data Management
- Real-time stock data from Yahoo Finance
- Customizable time periods (1 day to 10 years)
- Multiple interval options (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
- CSV export functionality
- Comparison of multiple stocks

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
cd stockapp
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main analysis with default settings (Apple stock):
```bash
python main.py
```

### Interactive Mode

For interactive stock analysis:

1. Uncomment the interactive_mode() line in `main.py`:
```python
if __name__ == "__main__":
    # Run main analysis
    # main()

    # Uncomment the line below to run in interactive mode instead
    interactive_mode()
```

2. Run the application:
```bash
python main.py
```

### Programmatic Usage

#### Basic Analysis
```python
from analyzer import StockAnalyzer

# Create analyzer for Apple stock
analyzer = StockAnalyzer("AAPL", period="1y", interval="1d")

# Run analysis
if analyzer.analyze():
    analyzer.print_analysis_summary()
    analyzer.plot_all_indicators()
```

#### Get Latest Analysis
```python
analysis = analyzer.get_latest_analysis()
print(f"Price: ${analysis['Close Price']:.2f}")
print(f"RSI: {analysis['RSI']}")
print(f"MACD: {analysis['MACD']}")
```

#### Get Trading Signals
```python
signals = analyzer.get_recent_signals(days=20)
print(signals)
```

#### Individual Indicator Plots
```python
analyzer.plot_rsi()
analyzer.plot_macd()
analyzer.plot_price_with_moving_averages()
analyzer.plot_bollinger_bands()
```

#### Export Data
```python
analyzer.export_analysis_to_csv("my_analysis.csv")
```

#### Compare Multiple Stocks
```python
from main import compare_stocks

compare_stocks(["AAPL", "MSFT", "GOOGL"], period="6mo")
```

## Configuration

Edit `config.py` to customize indicator parameters:

### RSI Settings
- `RSI_PERIOD`: Period for RSI calculation (default: 14)
- `RSI_OVERBOUGHT`: Overbought threshold (default: 70)
- `RSI_OVERSOLD`: Oversold threshold (default: 30)

### MACD Settings
- `MACD_FAST`: Fast EMA period (default: 12)
- `MACD_SLOW`: Slow EMA period (default: 26)
- `MACD_SIGNAL`: Signal line period (default: 9)

### Moving Averages
- `EMA_PERIODS`: Periods for exponential moving averages (default: [20, 50, 200])
- `SMA_PERIODS`: Periods for simple moving averages (default: [20, 50, 200])

### Bollinger Bands
- `BB_PERIOD`: Period for middle band SMA (default: 20)
- `BB_STD_DEV`: Number of standard deviations (default: 2)

### Chart Settings
- `CHART_FIGSIZE`: Figure size (default: (14, 10))
- `CHART_DPI`: Resolution (default: 100)
- `CHART_STYLE`: Matplotlib style (default: 'seaborn-v0_8-darkgrid')

## Indicator Explanations

### RSI (Relative Strength Index)
- **Range**: 0-100
- **Overbought**: > 70 (potential sell signal)
- **Oversold**: < 30 (potential buy signal)
- **Neutral**: 30-70

### MACD
- **Components**:
  - MACD Line: 12-day EMA - 26-day EMA
  - Signal Line: 9-day EMA of MACD line
  - Histogram: MACD Line - Signal Line
- **Buy Signal**: MACD crosses above signal line
- **Sell Signal**: MACD crosses below signal line

### Moving Averages
- **EMA 20**: Short-term trend
- **EMA 50**: Medium-term trend
- **EMA 200**: Long-term trend
- **Golden Cross**: EMA 50 crosses above EMA 200 (bullish)
- **Death Cross**: EMA 50 crosses below EMA 200 (bearish)

### Bollinger Bands
- **Upper Band**: SMA + (2 × Standard Deviation)
- **Middle Band**: SMA (20-day)
- **Lower Band**: SMA - (2 × Standard Deviation)
- **Buy Signal**: Price touches lower band
- **Sell Signal**: Price touches upper band

## Trading Signal Combinations

### Strong Buy Signal
- RSI < 30 (oversold)
- MACD > Signal Line (bullish)
- Price above EMA 200 (long-term uptrend)

### Strong Sell Signal
- RSI > 70 (overbought)
- MACD < Signal Line (bearish)
- Price below EMA 200 (long-term downtrend)

## File Structure

```
stockapp/
├── main.py                 # Main entry point and examples
├── analyzer.py            # Core analysis logic
├── data_fetcher.py        # Stock data retrieval
├── indicators.py          # Technical indicator calculations
├── visualizer.py          # Chart and visualization
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data Sources

- **Stock Data**: Yahoo Finance (via yfinance library)
- **Update Frequency**: Real-time (based on Yahoo Finance availability)

## Troubleshooting

### No data returned
- Verify the stock symbol is valid (use uppercase, e.g., "AAPL")
- Check internet connection
- Some symbols may have limited historical data

### Charts not displaying
- Ensure matplotlib is properly installed
- On Linux, may need to install additional dependencies
- Try using `plt.savefig()` instead of `plt.show()`

### Performance issues
- Reduce the time period for analysis
- Use longer intervals (e.g., "1wk" instead of "1d")
- Process fewer stocks at once

## Disclaimer

This application is for educational and informational purposes only. It should not be considered as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions.

Technical indicators are tools to help identify trends and potential entry/exit points, but they are not foolproof. No single indicator should be relied upon for trading decisions.

## License

This project is open source and available for personal and educational use.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bug reports and feature requests.

## Future Enhancements

- [ ] Additional indicators (Stochastic, AD, OBV)
- [ ] Portfolio tracking
- [ ] Backtesting functionality
- [ ] Advanced charting with custom timeframes
- [ ] Email alerts for trading signals
- [ ] Web dashboard interface
- [ ] Machine learning predictions
- [ ] Risk analysis tools
