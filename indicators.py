"""
Technical Indicators Module
Calculates MACD, RSI, and other technical indicators
"""

import pandas as pd
import numpy as np
import logging
from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_PERIODS, SMA_PERIODS, BB_PERIOD, BB_STD_DEV
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data

        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.logger = logger

    def calculate_rsi(self, period: int = RSI_PERIOD) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            period: Period for RSI calculation (default 14)

        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = self.data['Close'].diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gain and loss
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.data['RSI'] = rsi
        return rsi

    def calculate_macd(self, fast: int = MACD_FAST, slow: int = MACD_SLOW,
                       signal: int = MACD_SIGNAL) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Tuple of (MACD, Signal Line, Histogram)
        """
        # Calculate exponential moving averages
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # MACD Histogram
        histogram = macd_line - signal_line

        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram

        return macd_line, signal_line, histogram

    def calculate_ema(self, periods: list = EMA_PERIODS) -> None:
        """
        Calculate Exponential Moving Averages

        Args:
            periods: List of periods for EMA calculation
        """
        for period in periods:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()

    def calculate_sma(self, periods: list = SMA_PERIODS) -> None:
        """
        Calculate Simple Moving Averages

        Args:
            periods: List of periods for SMA calculation
        """
        for period in periods:
            self.data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()

    def calculate_bollinger_bands(self, period: int = BB_PERIOD,
                                 std_dev: int = BB_STD_DEV) -> tuple:
        """
        Calculate Bollinger Bands

        Args:
            period: Period for moving average (default 20)
            std_dev: Number of standard deviations (default 2)

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        self.data['BB_Upper'] = upper_band
        self.data['BB_Middle'] = sma
        self.data['BB_Lower'] = lower_band

        return upper_band, sma, lower_band

    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Returns:
            DataFrame with all indicators
        """
        self.logger.info("Calculating technical indicators...")

        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_ema()
        self.calculate_sma()
        self.calculate_bollinger_bands()

        self.logger.info("Technical indicators calculated successfully")
        return self.data

    def calculate_pivot_points(self) -> dict:
        """
        Calculate Pivot Points (Support and Resistance levels)
        Uses: P = (High + Low + Close) / 3
        R1 = (2 * P) - Low
        R2 = P + (High - Low)
        S1 = (2 * P) - High
        S2 = P - (High - Low)

        Returns:
            Dictionary with keys 'S2', 'S1', 'Pivot', 'R1', 'R2'
        """
        # Use the most recent candle (yesterday's H/L/C for today's levels)
        if len(self.data) < 1:
            return {}

        latest_idx = -1
        h = self.data['High'].iloc[latest_idx]
        l = self.data['Low'].iloc[latest_idx]
        c = self.data['Close'].iloc[latest_idx]

        # Calculate pivot point
        pivot = (h + l + c) / 3

        # Calculate resistance levels
        r1 = (2 * pivot) - l
        r2 = pivot + (h - l)

        # Calculate support levels
        s1 = (2 * pivot) - h
        s2 = pivot - (h - l)

        return {
            'S2': s2,
            'S1': s1,
            'Pivot': pivot,
            'R1': r1,
            'R2': r2
        }

    def get_sr_status(self, price: float, levels: dict) -> dict:
        """
        Determine S/R status for current price

        Args:
            price: Current price
            levels: Dictionary with S/R levels

        Returns:
            Dictionary with status info
        """
        if not levels:
            return {}

        status = {
            'current_price': price,
            'levels': levels,
            'nearest_support': None,
            'nearest_resistance': None,
            'zone': 'Unknown'
        }

        # Find nearest support (below price)
        supports = [('S2', levels['S2']), ('S1', levels['S1']), ('Pivot', levels['Pivot'])]
        for name, level in reversed(supports):
            if level < price:
                status['nearest_support'] = (name, level)
                break

        # Find nearest resistance (above price)
        resistances = [('Pivot', levels['Pivot']), ('R1', levels['R1']), ('R2', levels['R2'])]
        for name, level in resistances:
            if level > price:
                status['nearest_resistance'] = (name, level)
                break

        # Determine zone
        if price >= levels['R2']:
            status['zone'] = 'Above R2 (Strong Resistance)'
        elif price >= levels['R1']:
            status['zone'] = 'R1-R2 Zone (Resistance)'
        elif price >= levels['Pivot']:
            status['zone'] = 'Pivot-R1 Zone (Upper)'
        elif price >= levels['S1']:
            status['zone'] = 'S1-Pivot Zone (Lower)'
        elif price >= levels['S2']:
            status['zone'] = 'S2-S1 Zone (Support)'
        else:
            status['zone'] = 'Below S2 (Strong Support)'

        return status

    def get_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on indicators

        Returns:
            DataFrame with signal information
        """
        signals = pd.DataFrame(index=self.data.index)
        signals['Date'] = self.data.index
        signals['Close'] = self.data['Close']

        # RSI Signals
        signals['RSI'] = self.data['RSI']
        signals['RSI_Signal'] = 'NEUTRAL'
        signals.loc[self.data['RSI'] > RSI_OVERBOUGHT, 'RSI_Signal'] = 'OVERBOUGHT'
        signals.loc[self.data['RSI'] < RSI_OVERSOLD, 'RSI_Signal'] = 'OVERSOLD'

        # MACD Signals
        signals['MACD'] = self.data['MACD']
        signals['MACD_Signal'] = self.data['MACD_Signal']
        signals['MACD_Histogram'] = self.data['MACD_Histogram']

        # MACD Crossover Signal
        macd_above_signal = self.data['MACD'] > self.data['MACD_Signal']
        signals['MACD_Trend'] = 'BEARISH'
        signals.loc[macd_above_signal, 'MACD_Trend'] = 'BULLISH'

        # Combined Signal
        signals['Buy_Signal'] = 0
        signals['Sell_Signal'] = 0
        signals['Bullish_Signal'] = 0
        signals['Bearish_Signal'] = 0

        # Buy signal: RSI oversold AND MACD bullish crossover
        buy_condition = (
            (self.data['RSI'].fillna(50) < RSI_OVERSOLD) &
            (self.data['MACD'].fillna(0) > self.data['MACD_Signal'].fillna(0))
        )
        signals.loc[buy_condition, 'Buy_Signal'] = 1

        # Sell signal: RSI overbought AND MACD bearish crossover
        sell_condition = (
            (self.data['RSI'].fillna(50) > RSI_OVERBOUGHT) &
            (self.data['MACD'].fillna(0) < self.data['MACD_Signal'].fillna(0))
        )
        signals.loc[sell_condition, 'Sell_Signal'] = 1

        # Bullish Signal: MACD bullish AND RSI shows positive momentum (40-70 range is healthy)
        bullish_condition = (
            (self.data['MACD'].fillna(0) > self.data['MACD_Signal'].fillna(0)) &  # MACD bullish
            (self.data['RSI'].fillna(50) > 40) &  # Not oversold, showing momentum
            (self.data['RSI'].fillna(50) <= RSI_OVERBOUGHT)  # Not overbought yet
        )
        signals.loc[bullish_condition, 'Bullish_Signal'] = 1

        # Bearish Signal: MACD bearish AND RSI shows negative momentum (30-60 range is healthy for bearish)
        bearish_condition = (
            (self.data['MACD'].fillna(0) < self.data['MACD_Signal'].fillna(0)) &  # MACD bearish
            (self.data['RSI'].fillna(50) < 60) &  # Not overbought, showing downward momentum
            (self.data['RSI'].fillna(50) >= RSI_OVERSOLD)  # Not oversold yet
        )
        signals.loc[bearish_condition, 'Bearish_Signal'] = 1

        return signals
