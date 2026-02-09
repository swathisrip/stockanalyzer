"""
Visualization Module
Creates charts for stock analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import logging
from config import CHART_FIGSIZE, CHART_DPI, CHART_STYLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockVisualizer:
    """Create visualizations for stock analysis"""

    def __init__(self, data: pd.DataFrame, symbol: str):
        """
        Initialize visualizer

        Args:
            data: DataFrame with OHLCV and indicator data
            symbol: Stock symbol
        """
        self.data = data
        self.symbol = symbol
        plt.style.use(CHART_STYLE)

    def plot_price_with_moving_averages(self, save_path: str = None) -> None:
        """
        Plot price with moving averages

        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)

        # Plot closing price
        ax.plot(self.data.index, self.data['Close'], label='Close Price',
                color='black', linewidth=2, zorder=3)

        # Plot EMAs
        if 'EMA_20' in self.data.columns:
            ax.plot(self.data.index, self.data['EMA_20'], label='EMA 20',
                    color='blue', linewidth=1.5, alpha=0.7)
        if 'EMA_50' in self.data.columns:
            ax.plot(self.data.index, self.data['EMA_50'], label='EMA 50',
                    color='green', linewidth=1.5, alpha=0.7)
        if 'EMA_200' in self.data.columns:
            ax.plot(self.data.index, self.data['EMA_200'], label='EMA 200',
                    color='red', linewidth=1.5, alpha=0.7)

        ax.set_title(f'{self.symbol} Price with Moving Averages', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI)
            logger.info(f"Chart saved to {save_path}")

        plt.show()

    def plot_rsi(self, save_path: str = None) -> None:
        """
        Plot RSI indicator

        Args:
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=CHART_FIGSIZE, dpi=CHART_DPI,
                                        gridspec_kw={'height_ratios': [1, 2]})

        # Plot price
        ax1.plot(self.data.index, self.data['Close'], label='Close Price',
                color='black', linewidth=2)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{self.symbol} Price and RSI', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot RSI
        ax2.plot(self.data.index, self.data['RSI'], label='RSI (14)',
                color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)',
                   linewidth=1)
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)',
                   linewidth=1)
        ax2.fill_between(self.data.index, 70, 100, alpha=0.2, color='red')
        ax2.fill_between(self.data.index, 0, 30, alpha=0.2, color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.set_ylim([0, 100])
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI)
            logger.info(f"RSI chart saved to {save_path}")

        plt.show()

    def plot_macd(self, save_path: str = None) -> None:
        """
        Plot MACD indicator

        Args:
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=CHART_FIGSIZE, dpi=CHART_DPI,
                                        gridspec_kw={'height_ratios': [1, 2]})

        # Plot price
        ax1.plot(self.data.index, self.data['Close'], label='Close Price',
                color='black', linewidth=2)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{self.symbol} Price and MACD', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot MACD
        ax2.plot(self.data.index, self.data['MACD'], label='MACD (12, 26)',
                color='blue', linewidth=2)
        ax2.plot(self.data.index, self.data['MACD_Signal'], label='Signal (9)',
                color='red', linewidth=2)

        # Plot histogram
        colors = ['green' if val > 0 else 'red' for val in self.data['MACD_Histogram']]
        ax2.bar(self.data.index, self.data['MACD_Histogram'], label='Histogram',
               color=colors, alpha=0.3, width=1)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI)
            logger.info(f"MACD chart saved to {save_path}")

        plt.show()

    def plot_bollinger_bands(self, save_path: str = None) -> None:
        """
        Plot Bollinger Bands

        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)

        # Plot price
        ax.plot(self.data.index, self.data['Close'], label='Close Price',
                color='black', linewidth=2, zorder=3)

        # Plot middle band
        ax.plot(self.data.index, self.data['BB_Middle'], label='Middle Band (SMA 20)',
                color='blue', linewidth=1.5, linestyle='--', alpha=0.7)

        # Plot upper and lower bands
        ax.plot(self.data.index, self.data['BB_Upper'], label='Upper Band',
                color='red', linewidth=1.5, alpha=0.7)
        ax.plot(self.data.index, self.data['BB_Lower'], label='Lower Band',
                color='green', linewidth=1.5, alpha=0.7)

        # Fill area between bands
        ax.fill_between(self.data.index, self.data['BB_Upper'],
                       self.data['BB_Lower'], alpha=0.1, color='gray')

        ax.set_title(f'{self.symbol} Price with Bollinger Bands', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI)
            logger.info(f"Bollinger Bands chart saved to {save_path}")

        plt.show()

    def plot_all_indicators(self, save_path: str = None) -> None:
        """
        Plot all indicators in one comprehensive view

        Args:
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=CHART_DPI)
        gs = fig.add_gridspec(4, 1, hspace=0.4)

        # Price and moving averages
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='Close Price',
                color='black', linewidth=2, zorder=3)
        if 'EMA_20' in self.data.columns:
            ax1.plot(self.data.index, self.data['EMA_20'], label='EMA 20',
                    color='blue', alpha=0.7)
        if 'EMA_50' in self.data.columns:
            ax1.plot(self.data.index, self.data['EMA_50'], label='EMA 50',
                    color='green', alpha=0.7)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{self.symbol} Technical Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # RSI
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.data.index, self.data['RSI'], color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(self.data.index, 70, 100, alpha=0.1, color='red')
        ax2.fill_between(self.data.index, 0, 30, alpha=0.1, color='green')
        ax2.set_ylabel('RSI')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)
        ax2.legend(['RSI (14)', 'Overbought', 'Oversold'], loc='upper left')

        # MACD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue', linewidth=2)
        ax3.plot(self.data.index, self.data['MACD_Signal'], label='Signal', color='red', linewidth=2)
        colors = ['green' if val > 0 else 'red' for val in self.data['MACD_Histogram']]
        ax3.bar(self.data.index, self.data['MACD_Histogram'], color=colors, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('MACD')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')

        # Volume
        ax4 = fig.add_subplot(gs[3])
        if 'Volume' in self.data.columns:
            colors = ['green' if self.data['Close'].iloc[i] >= self.data['Open'].iloc[i]
                     else 'red' for i in range(len(self.data))]
            ax4.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.6)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volume')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI)
            logger.info(f"Comprehensive chart saved to {save_path}")

        plt.show()


class ChartHelper:
    """Helper functions for creating charts"""

    @staticmethod
    def save_figure(fig, path: str) -> None:
        """Save figure to file"""
        fig.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {path}")
