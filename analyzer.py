"""
Stock Analyzer Module
Main analysis logic combining data, indicators, and signals
"""

import pandas as pd
import logging
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from visualizer import StockVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    """Complete stock analysis system"""

    def __init__(self, symbol: str, period: str = "1y", interval: str = "1d"):
        """
        Initialize Stock Analyzer

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period
            interval: Data interval
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data_fetcher = DataFetcher(symbol)
        self.indicators = None
        self.signals = None
        self.data = None
        self.visualizer = None

    def fetch_data(self) -> bool:
        """Fetch stock data"""
        logger.info(f"Fetching data for {self.symbol}...")
        self.data = self.data_fetcher.fetch_data(self.period, self.interval)

        if self.data is None or self.data.empty:
            logger.error(f"Failed to fetch data for {self.symbol}")
            return False

        logger.info(f"Successfully fetched {len(self.data)} records")
        return True

    def calculate_indicators(self) -> bool:
        """Calculate technical indicators"""
        if self.data is None or self.data.empty:
            logger.error("No data available. Please fetch data first.")
            return False

        logger.info("Calculating technical indicators...")
        self.indicators = TechnicalIndicators(self.data)
        self.data = self.indicators.calculate_all_indicators()

        logger.info("Technical indicators calculated successfully")
        return True

    def generate_signals(self) -> bool:
        """Generate trading signals"""
        if self.indicators is None:
            logger.error("Indicators not calculated. Please run calculate_indicators() first.")
            return False

        logger.info("Generating trading signals...")
        self.signals = self.indicators.get_signals()
        logger.info("Trading signals generated successfully")
        return True

    def analyze(self) -> bool:
        """Run complete analysis pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting analysis for {self.symbol}")
        logger.info(f"{'='*60}\n")

        # Fetch data
        if not self.fetch_data():
            return False

        # Calculate indicators
        if not self.calculate_indicators():
            return False

        # Generate signals
        if not self.generate_signals():
            return False

        # Initialize visualizer
        self.visualizer = StockVisualizer(self.data, self.symbol)

        logger.info("Analysis completed successfully")
        return True

    def get_latest_analysis(self) -> dict:
        """Get latest analysis results"""
        if self.data is None or self.data.empty:
            return None

        latest_idx = -1
        latest_date = self.data.index[latest_idx]
        latest_price = self.data['Close'].iloc[latest_idx]
        latest_rsi = self.data['RSI'].iloc[latest_idx]
        latest_macd = self.data['MACD'].iloc[latest_idx]
        latest_signal = self.data['MACD_Signal'].iloc[latest_idx]

        analysis = {
            'Date': latest_date,
            'Symbol': self.symbol,
            'Close Price': latest_price,
            'RSI': round(latest_rsi, 2) if pd.notna(latest_rsi) else None,
            'MACD': round(latest_macd, 4) if pd.notna(latest_macd) else None,
            'MACD Signal': round(latest_signal, 4) if pd.notna(latest_signal) else None,
            'MACD Histogram': round(self.data['MACD_Histogram'].iloc[latest_idx], 4)
            if pd.notna(self.data['MACD_Histogram'].iloc[latest_idx]) else None,
        }

        return analysis

    def print_analysis_summary(self) -> None:
        """Print analysis summary to console"""
        if self.data is None or self.data.empty:
            print("No data available")
            return

        analysis = self.get_latest_analysis()

        print(f"\n{'='*60}")
        print(f"STOCK ANALYSIS SUMMARY: {self.symbol}")
        print(f"{'='*60}")
        print(f"Date: {analysis['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Close Price: ${analysis['Close Price']:.2f}")

        # Price change
        change, pct_change = self.data_fetcher.get_price_change()
        if change is not None:
            change_emoji = "📈" if change > 0 else "📉"
            print(f"Period Change: {change_emoji} ${change:.2f} ({pct_change:.2f}%)")

        print(f"\n{'Technical Indicators:':-^60}")
        print(f"RSI (14): {analysis['RSI']}")
        if analysis['RSI'] is not None:
            if analysis['RSI'] > 70:
                print("  Status: OVERBOUGHT ⚠️")
            elif analysis['RSI'] < 30:
                print("  Status: OVERSOLD ✓")
            else:
                print("  Status: NORMAL")

        print(f"\nMACD (12, 26, 9):")
        print(f"  MACD Line: {analysis['MACD']}")
        print(f"  Signal Line: {analysis['MACD Signal']}")
        print(f"  Histogram: {analysis['MACD Histogram']}")

        if analysis['MACD'] is not None and analysis['MACD Signal'] is not None:
            if analysis['MACD'] > analysis['MACD Signal']:
                print("  Trend: BULLISH 📈")
            else:
                print("  Trend: BEARISH 📉")

        # Get latest signals if available
        if self.signals is not None and len(self.signals) > 0:
            latest_signal_row = self.signals.iloc[-1]
            print(f"\n{'Trading Signals:':-^60}")
            if latest_signal_row['Buy_Signal'] == 1:
                print("⬆️  BUY SIGNAL DETECTED (RSI Oversold + MACD Bullish)")
            elif latest_signal_row['Sell_Signal'] == 1:
                print("⬇️  SELL SIGNAL DETECTED (RSI Overbought + MACD Bearish)")
            elif latest_signal_row['Bullish_Signal'] == 1:
                print("📈 BULLISH SIGNAL (MACD Bullish + RSI Momentum)")
            elif latest_signal_row['Bearish_Signal'] == 1:
                print("📉 BEARISH SIGNAL (MACD Bearish + RSI Decline)")
            else:
                print("No active signals")

        print(f"\n{'='*60}\n")

    def get_recent_signals(self, days: int = 10) -> pd.DataFrame:
        """Get recent trading signals"""
        if self.signals is None:
            return None

        recent = self.signals.tail(days)
        buy_signals = recent[recent['Buy_Signal'] == 1]
        sell_signals = recent[recent['Sell_Signal'] == 1]
        bullish_signals = recent[recent['Bullish_Signal'] == 1]
        bearish_signals = recent[recent['Bearish_Signal'] == 1]

        result = []

        for _, row in buy_signals.iterrows():
            result.append({
                'Date': row['Date'],
                'Close': row['Close'],
                'Signal': 'BUY 🔼',
                'Type': 'RSI Oversold + MACD Bullish'
            })

        for _, row in sell_signals.iterrows():
            result.append({
                'Date': row['Date'],
                'Close': row['Close'],
                'Signal': 'SELL 🔽',
                'Type': 'RSI Overbought + MACD Bearish'
            })

        for _, row in bullish_signals.iterrows():
            result.append({
                'Date': row['Date'],
                'Close': row['Close'],
                'Signal': 'BULLISH 📈',
                'Type': 'MACD Bullish + RSI Momentum'
            })

        for _, row in bearish_signals.iterrows():
            result.append({
                'Date': row['Date'],
                'Close': row['Close'],
                'Signal': 'BEARISH 📉',
                'Type': 'MACD Bearish + RSI Decline'
            })

        if result:
            result_df = pd.DataFrame(result)
            return result_df.sort_values('Date')
        return pd.DataFrame()

    def plot_rsi(self, save_path: str = None) -> None:
        """Plot RSI chart"""
        if self.visualizer is None:
            logger.error("Please run analyze() first")
            return
        self.visualizer.plot_rsi(save_path)

    def plot_macd(self, save_path: str = None) -> None:
        """Plot MACD chart"""
        if self.visualizer is None:
            logger.error("Please run analyze() first")
            return
        self.visualizer.plot_macd(save_path)

    def plot_price_with_moving_averages(self, save_path: str = None) -> None:
        """Plot price with moving averages"""
        if self.visualizer is None:
            logger.error("Please run analyze() first")
            return
        self.visualizer.plot_price_with_moving_averages(save_path)

    def plot_bollinger_bands(self, save_path: str = None) -> None:
        """Plot Bollinger Bands"""
        if self.visualizer is None:
            logger.error("Please run analyze() first")
            return
        self.visualizer.plot_bollinger_bands(save_path)

    def plot_all_indicators(self, save_path: str = None) -> None:
        """Plot comprehensive view with all indicators"""
        if self.visualizer is None:
            logger.error("Please run analyze() first")
            return
        self.visualizer.plot_all_indicators(save_path)

    def export_analysis_to_csv(self, filename: str = None) -> None:
        """Export analysis data to CSV"""
        if self.data is None:
            logger.error("No data to export")
            return

        if filename is None:
            filename = f"{self.symbol}_analysis.csv"

        self.data.to_csv(filename)
        logger.info(f"Analysis exported to {filename}")
