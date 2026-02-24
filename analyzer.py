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
        latest_hma = self.data['HMA'].iloc[latest_idx] if 'HMA' in self.data.columns else None
        
        # Determine current MACD trend (not historical signals)
        if pd.notna(latest_macd) and pd.notna(latest_signal) and latest_macd is not None and latest_signal is not None:
            try:
                macd_trend = 'BULLISH' if latest_macd > latest_signal else 'BEARISH'
            except (TypeError, ValueError):
                macd_trend = 'N/A'
        else:
            macd_trend = 'N/A'
        
        # Determine RSI state
        if pd.notna(latest_rsi) and latest_rsi is not None:
            try:
                if latest_rsi > 70:
                    rsi_state = 'OVERBOUGHT'
                elif latest_rsi < 30:
                    rsi_state = 'OVERSOLD'
                else:
                    rsi_state = 'NEUTRAL'
            except (TypeError, ValueError):
                rsi_state = 'N/A'
        else:
            rsi_state = 'N/A'
        
        # Determine HMA trend (bullish if above price, bearish if below)
        if pd.notna(latest_hma) and pd.notna(latest_price) and latest_hma is not None and latest_price is not None:
            try:
                hma_trend = 'BULLISH' if latest_hma < latest_price else 'BEARISH'
            except (TypeError, ValueError):
                hma_trend = 'N/A'
        else:
            hma_trend = 'N/A'

        analysis = {
            'Date': latest_date,
            'Symbol': self.symbol,
            'Close Price': latest_price,
            'RSI': round(latest_rsi, 2) if pd.notna(latest_rsi) else None,
            'RSI State': rsi_state,
            'MACD': round(latest_macd, 4) if pd.notna(latest_macd) else None,
            'MACD Signal': round(latest_signal, 4) if pd.notna(latest_signal) else None,
            'MACD Histogram': round(self.data['MACD_Histogram'].iloc[latest_idx], 4)
            if pd.notna(self.data['MACD_Histogram'].iloc[latest_idx]) else None,
            'MACD Trend (Current)': macd_trend,
            'HMA': round(latest_hma, 2) if pd.notna(latest_hma) else None,
            'HMA Trend': hma_trend,
        }

        return analysis

    def get_macd_analysis_at_date(self, date_str: str = None) -> dict:
        """Get detailed MACD analysis for a specific date"""
        if self.data is None or self.data.empty:
            return None
        
        if date_str:
            # Find row by date
            try:
                date_obj = pd.to_datetime(date_str)
                if date_obj in self.data.index:
                    row = self.data.loc[date_obj]
                else:
                    # Find closest date
                    idx = self.data.index.get_indexer([date_obj], method='nearest')[0]
                    row = self.data.iloc[idx]
            except:
                # If date not found, use latest
                row = self.data.iloc[-1]
        else:
            # Use latest
            row = self.data.iloc[-1]
        
        macd_val = row['MACD']
        signal_val = row['MACD_Signal']
        histogram_val = row['MACD_Histogram']
        
        # Determine trend
        if pd.notna(macd_val) and pd.notna(signal_val):
            is_bullish = macd_val > signal_val
            trend = 'BULLISH' if is_bullish else 'BEARISH'
        else:
            trend = 'N/A'
        
        return {
            'Date': row['Date'] if 'Date' in row else row.name,
            'MACD': round(macd_val, 6) if pd.notna(macd_val) else None,
            'Signal': round(signal_val, 6) if pd.notna(signal_val) else None,
            'Histogram': round(histogram_val, 6) if pd.notna(histogram_val) else None,
            'MACD > Signal': macd_val > signal_val if pd.notna(macd_val) and pd.notna(signal_val) else None,
            'Trend': trend,
            'Raw_MACD': macd_val,
            'Raw_Signal': signal_val,
            'Comparison': f'MACD ({macd_val:.6f}) vs Signal ({signal_val:.6f})'
        }
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

    def get_macd_rsi_signals(self, count: int = 2) -> pd.DataFrame:
        """Get recent MACD + RSI signals"""
        if self.signals is None:
            return pd.DataFrame()
        
        signals = self.signals.copy()
        buy_sell = signals[(signals['Buy_Signal'] == 1) | (signals['Sell_Signal'] == 1)]
        result = []
        
        for _, row in buy_sell.iterrows():
            signal_type = 'BUY 🔼' if row['Buy_Signal'] == 1 else 'SELL 🔽'
            result.append({
                'Date': row['Date'],
                'Close': f"${row['Close']:.2f}",
                'Signal': signal_type,
                'Type': 'MACD + RSI'
            })
        
        if result:
            result_df = pd.DataFrame(result).sort_values('Date', ascending=False).head(count)
            return result_df.reset_index(drop=True)
        return pd.DataFrame()

    def get_hma_signals(self, count: int = 2) -> pd.DataFrame:
        """Get recent HMA signals"""
        if self.signals is None:
            return pd.DataFrame()
        
        signals = self.signals.copy()
        hma_signals = signals[(signals['HMA_Buy_Signal'] == 1) | (signals['HMA_Sell_Signal'] == 1)]
        result = []
        
        for _, row in hma_signals.iterrows():
            signal_type = 'BUY 🔼' if row['HMA_Buy_Signal'] == 1 else 'SELL 🔽'
            result.append({
                'Date': row['Date'],
                'Close': f"${row['Close']:.2f}",
                'Signal': signal_type,
                'Type': 'HMA'
            })
        
        if result:
            result_df = pd.DataFrame(result).sort_values('Date', ascending=False).head(count)
            return result_df.reset_index(drop=True)
        return pd.DataFrame()

    def get_macd_hma_signals(self, count: int = 2) -> pd.DataFrame:
        """Get recent MACD + HMA signals (agreement signals)"""
        if self.signals is None:
            return pd.DataFrame()
        
        signals = self.signals.copy()
        macd_hma = signals[(signals['MACD_HMA_Buy_Signal'] == 1) | (signals['MACD_HMA_Sell_Signal'] == 1)]
        result = []
        
        for _, row in macd_hma.iterrows():
            signal_type = 'BUY 🔼' if row['MACD_HMA_Buy_Signal'] == 1 else 'SELL 🔽'
            result.append({
                'Date': row['Date'],
                'Close': f"${row['Close']:.2f}",
                'Signal': signal_type,
                'Type': 'MACD + HMA'
            })
        
        if result:
            result_df = pd.DataFrame(result).sort_values('Date', ascending=False).head(count)
            return result_df.reset_index(drop=True)
        return pd.DataFrame()

    def get_macd_hma_divergence_signals(self, count: int = 1) -> pd.DataFrame:
        """Get MACD + HMA divergence signals (disagreement = potential reversals)
        Only shows divergences from the most recent date to ensure latest MACD state"""
        if self.signals is None or self.data is None:
            return pd.DataFrame()
        
        signals = self.signals.copy()
        
        # Get the latest date in the data
        if len(signals) == 0:
            return pd.DataFrame()
        
        latest_date = signals['Date'].max()
        
        # Filter for recent divergences (from latest date or recent dates only)
        # This ensures we're using the latest MACD state, not historical ones
        divergence = signals[(signals['MACD_HMA_Divergence_BearishBuy'] == 1) | (signals['MACD_HMA_Divergence_BullishSell'] == 1)]
        
        # Further filter: only show divergences from the latest date
        if len(divergence) > 0:
            divergence = divergence[divergence['Date'] == latest_date]
        
        result = []
        
        for _, row in divergence.iterrows():
            if row['MACD_HMA_Divergence_BearishBuy'] == 1:
                signal_type = '⚠️ MACD Bearish + HMA Buy'
                desc = 'Bullish Divergence'
            else:
                signal_type = '⚠️ MACD Bullish + HMA Sell'
                desc = 'Bearish Divergence'
            
            result.append({
                'Date': row['Date'],
                'Close': f"${row['Close']:.2f}",
                'Signal': signal_type,
                'Type': desc
            })
        
        if result:
            result_df = pd.DataFrame(result).sort_values('Date', ascending=False).head(count)
            return result_df.reset_index(drop=True)
        return pd.DataFrame()

    def get_macd_crossovers(self, count: int = 5) -> pd.DataFrame:
        """Get recent MACD crossover points (where trend changed)"""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        data = self.data.copy()
        
        # Detect crossovers: current MACD crosses signal line
        macd_above = (data['MACD'] > data['MACD_Signal']).astype(int)
        crossover = macd_above.diff().fillna(0) != 0  # True when trend changes
        
        crossover_data = data[crossover].copy()
        
        result = []
        for _, row in crossover_data.iterrows():
            macd_val = row['MACD']
            signal_val = row['MACD_Signal']
            trend = 'BULLISH ↑' if macd_val > signal_val else 'BEARISH ↓'
            
            result.append({
                'Date': row.name,
                'Close': f"${row['Close']:.2f}",
                'Trend': trend,
                'MACD': f"{macd_val:.4f}",
                'Signal': f"{signal_val:.4f}"
            })
        
        if result:
            result_df = pd.DataFrame(result).sort_values('Date', ascending=False).head(count)
            return result_df.reset_index(drop=True)
        return pd.DataFrame()

    def get_combined_signals(self, count: int = 2) -> pd.DataFrame:
        """Get combined signals based on latest state of each indicator (MACD + RSI + HMA)"""
        if self.data is None or self.data.empty or self.signals is None:
            return pd.DataFrame()
        
        data = self.data.copy()
        signals = self.signals.copy()
        result = []
        
        # Check latest state of each indicator
        latest_date = data.index[-1]
        latest_row = data.iloc[-1]
        latest_signal_row = signals.iloc[-1]
        
        # Get indicator states
        macd_val = latest_row['MACD']
        signal_val = latest_row['MACD_Signal']
        rsi_val = latest_row['RSI']
        hma_bullish = latest_signal_row['HMA_Buy_Signal'] == 1 if pd.notna(latest_signal_row['HMA_Buy_Signal']) else False
        
        # Check if each indicator is bullish/bearish
        macd_bullish = pd.notna(macd_val) and pd.notna(signal_val) and macd_val > signal_val
        rsi_bullish = pd.notna(rsi_val) and rsi_val > 50
        
        # Count bullish indicators
        bullish_count = sum([macd_bullish, rsi_bullish, hma_bullish])
        
        # Determine combined signal
        if bullish_count >= 2:
            signal_type = 'BUY 🔼'
            agreement = f"{bullish_count}/3 bullish"
        elif bullish_count == 0:
            signal_type = 'SELL 🔽'
            agreement = "0/3 bullish"
        else:
            signal_type = 'MIXED ⚖️'
            agreement = f"{bullish_count}/3 bullish"
        
        result.append({
            'Date': latest_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest_date, 'strftime') else str(latest_date),
            'Close': f"${latest_row['Close']:.2f}",
            'Signal': signal_type,
            'Type': agreement
        })
        
        if result:
            return pd.DataFrame(result).reset_index(drop=True)
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
