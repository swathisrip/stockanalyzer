"""
Web Visualizer Module
Creates interactive Plotly visualizations for the web app
"""

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebVisualizer:
    """Create interactive Plotly charts for web visualization"""

    def __init__(self, data: pd.DataFrame, symbol: str):
        """
        Initialize WebVisualizer

        Args:
            data: DataFrame with OHLCV data and indicators
            symbol: Stock symbol
        """
        self.data = data
        self.symbol = symbol

    def plot_price_with_moving_averages(self):
        """Plot price with EMA/SMA overlays using candlestick chart"""
        fig = go.Figure()

        # Add candlestick chart (Open, High, Low, Close)
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='Price Action'
        ))

        # Add EMAs
        if 'EMA_20' in self.data.columns and self.data['EMA_20'].notna().any():
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['EMA_20'],
                name='EMA 20',
                mode='lines',
                line=dict(color='#ff7f0e', width=1, dash='dash')
            ))

        if 'EMA_50' in self.data.columns and self.data['EMA_50'].notna().any():
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['EMA_50'],
                name='EMA 50',
                mode='lines',
                line=dict(color='#2ca02c', width=1, dash='dash')
            ))

        if 'EMA_200' in self.data.columns and self.data['EMA_200'].notna().any():
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['EMA_200'],
                name='EMA 200',
                mode='lines',
                line=dict(color='#d62728', width=1, dash='dash')
            ))

        # Add SMAs
        if 'SMA_20' in self.data.columns and self.data['SMA_20'].notna().any():
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['SMA_20'],
                name='SMA 20',
                mode='lines',
                line=dict(color='#9467bd', width=1, dash='dot'),
                visible='legendonly'
            ))

        fig.update_layout(
            title=f"{self.symbol} - Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        return fig

    def plot_rsi(self):
        """Plot RSI with overbought/oversold zones"""
        fig = go.Figure()

        # Add RSI line
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['RSI'],
            name='RSI (14)',
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Add overbought line (70)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            annotation_text="Overbought (70)",
            annotation_position="right"
        )

        # Add oversold line (30)
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Oversold (30)",
            annotation_position="right"
        )

        # Add shaded regions
        fig.add_hrect(
            y0=70, y1=100,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0
        )

        fig.add_hrect(
            y0=0, y1=30,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0
        )

        fig.update_layout(
            title=f"{self.symbol} - RSI (Relative Strength Index)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        return fig

    def plot_macd(self):
        """Plot MACD with signal line and histogram"""
        fig = go.Figure()

        # Add histogram (MACD - Signal)
        colors = ['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['MACD_Histogram'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.3
        ))

        # Add MACD line
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['MACD'],
            name='MACD Line',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add Signal line
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['MACD_Signal'],
            name='Signal Line',
            mode='lines',
            line=dict(color='#ff7f0e', width=2)
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", layer="below")

        fig.update_layout(
            title=f"{self.symbol} - MACD (Moving Average Convergence Divergence)",
            xaxis_title="Date",
            yaxis_title="MACD Value",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        return fig

    def plot_bollinger_bands(self):
        """Plot Bollinger Bands"""
        fig = go.Figure()

        # Add upper band
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['BB_Upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='Upper Band (SMA + 2σ)',
            showlegend=True
        ))

        # Add lower band and fill between
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['BB_Lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='Lower Band (SMA - 2σ)',
            fillcolor='rgba(255,0,0,0.1)'
        ))

        # Add middle band (SMA)
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['BB_Middle'],
            name='Middle Band (SMA 20)',
            mode='lines',
            line=dict(color='gray', width=1, dash='dash')
        ))

        # Add closing price
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close'],
            name='Close Price',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title=f"{self.symbol} - Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        return fig

    def plot_all_indicators(self):
        """Plot all indicators in a 2x2 subplot grid"""
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Price with Moving Averages",
                "RSI (Relative Strength Index)",
                "MACD (Moving Average Convergence Divergence)",
                "Bollinger Bands"
            ),
            specs=[[{}, {}], [{}, {}]]
        )

        # 1. Price with Moving Averages (top-left) - using candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price Action',
                showlegend=True
            ),
            row=1, col=1
        )

        if 'EMA_20' in self.data.columns and self.data['EMA_20'].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['EMA_20'],
                    name='EMA 20',
                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )

        if 'EMA_50' in self.data.columns and self.data['EMA_50'].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['EMA_50'],
                    name='EMA 50',
                    line=dict(color='#2ca02c', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )

        # 2. RSI (top-right)
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['RSI'],
                name='RSI',
                line=dict(color='#1f77b4', width=1),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                showlegend=False
            ),
            row=1, col=2
        )

        # Add overbought/oversold lines to RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)

        # 3. MACD (bottom-left)
        colors = ['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.3,
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MACD'],
                name='MACD',
                line=dict(color='#1f77b4', width=1),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MACD_Signal'],
                name='Signal',
                line=dict(color='#ff7f0e', width=1),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Bollinger Bands (bottom-right)
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['BB_Upper'],
                fill=None,
                line_color='rgba(255,0,0,0.2)',
                name='Upper Band',
                showlegend=False
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['BB_Lower'],
                fill='tonexty',
                line_color='rgba(255,0,0,0.2)',
                fillcolor='rgba(255,0,0,0.1)',
                name='Lower Band',
                showlegend=False
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                name='Price',
                line=dict(color='#1f77b4', width=1),
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=1, col=2)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=2, col=2)

        fig.update_layout(
            title_text=f"{self.symbol} - Technical Indicators Dashboard",
            height=1000,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_price_with_sr_levels(self, pivot_levels: dict):
        """Plot price with Support/Resistance (Pivot Point) levels"""
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close'],
            name='Close Price',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))

        if not pivot_levels:
            return fig

        # Add R2 (Strong Resistance)
        fig.add_hline(
            y=pivot_levels['R2'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"R2: ${pivot_levels['R2']:.2f}",
            annotation_position="right",
            name="R2"
        )

        # Add R1 (Resistance)
        fig.add_hline(
            y=pivot_levels['R1'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"R1: ${pivot_levels['R1']:.2f}",
            annotation_position="right",
            name="R1"
        )

        # Add Pivot Point
        fig.add_hline(
            y=pivot_levels['Pivot'],
            line_dash="solid",
            line_color="gray",
            annotation_text=f"Pivot: ${pivot_levels['Pivot']:.2f}",
            annotation_position="right",
            name="Pivot",
            line=dict(width=2)
        )

        # Add S1 (Support)
        fig.add_hline(
            y=pivot_levels['S1'],
            line_dash="dash",
            line_color="lightgreen",
            annotation_text=f"S1: ${pivot_levels['S1']:.2f}",
            annotation_position="right",
            name="S1"
        )

        # Add S2 (Strong Support)
        fig.add_hline(
            y=pivot_levels['S2'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"S2: ${pivot_levels['S2']:.2f}",
            annotation_position="right",
            name="S2"
        )

        # Add shaded zones
        # R1-R2 Zone (Resistance)
        fig.add_hrect(
            y0=pivot_levels['R1'], y1=pivot_levels['R2'],
            fillcolor="red",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        # Pivot-R1 Zone (Upper)
        fig.add_hrect(
            y0=pivot_levels['Pivot'], y1=pivot_levels['R1'],
            fillcolor="orange",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        # S1-Pivot Zone (Lower)
        fig.add_hrect(
            y0=pivot_levels['S1'], y1=pivot_levels['Pivot'],
            fillcolor="lightblue",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        # S2-S1 Zone (Support)
        fig.add_hrect(
            y0=pivot_levels['S2'], y1=pivot_levels['S1'],
            fillcolor="green",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        fig.update_layout(
            title=f"{self.symbol} - Price with Support/Resistance (Pivot Points)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )

        return fig

    def plot_signals_timeline(self, signals_df):
        """Plot trading signals on a timeline"""
        fig = go.Figure()

        # Add price line as background
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close'],
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))

        # Define signal colors and markers
        signal_colors = {
            'BUY 🔼': '#00cc00',
            'SELL 🔽': '#ff0000',
            'BULLISH 📈': '#90EE90',
            'BEARISH 📉': '#FFB6C6'
        }

        # Add signals as scatter points
        for signal_type in ['BUY 🔼', 'SELL 🔽', 'BULLISH 📈', 'BEARISH 📉']:
            signal_data = signals_df[signals_df['Signal'] == signal_type]
            if len(signal_data) > 0:
                fig.add_trace(go.Scatter(
                    x=signal_data['Date'],
                    y=signal_data['Close'],
                    mode='markers',
                    name=signal_type,
                    marker=dict(
                        size=10,
                        color=signal_colors.get(signal_type, '#000000'),
                        symbol='diamond'
                    ),
                    text=signal_data['Type'],
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>%{text}<extra></extra>'
                ))

        fig.update_layout(
            title=f"{self.symbol} - Trading Signals Timeline",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_market_indices(indices_data: dict):
        """
        Plot market indices intraday performance with normalized returns

        Args:
            indices_data: Dictionary from MarketFetcher.fetch_market_indices()
                Contains data for SPY, QQQ, DIA, IWM, VIX

        Returns:
            Plotly figure with multi-line chart
        """
        fig = go.Figure()

        # Define colors for each index
        index_colors = {
            'SPY': '#1f77b4',   # Blue - S&P 500
            'QQQ': '#ff7f0e',   # Orange - Nasdaq-100
            'DIA': '#2ca02c',   # Green - Dow Jones
            'IWM': '#d62728',   # Red - Russell 2000
            'VIX': '#9467bd'    # Purple - Volatility
        }

        # Add a line for each index
        for symbol, data in indices_data.items():
            if symbol not in index_colors:
                continue

            # Skip if data is 'N/A'
            if isinstance(data.get('price'), str) and data['price'] == 'N/A':
                continue

            # Extract price change percentage
            change_pct = data.get('change_percent', 0)
            if isinstance(change_pct, str):
                continue

            # Add trace for this index
            fig.add_trace(go.Scatter(
                x=[data['name']],
                y=[change_pct],
                name=f"{symbol} ({data['name']})",
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=index_colors[symbol],
                    line=dict(color='white', width=2)
                ),
                text=[f"{change_pct:+.2f}%"],
                textposition='top center',
                textfont=dict(size=12, color=index_colors[symbol]),
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    f"Name: {data['name']}<br>"
                    f"Price: ${data['price']}<br>"
                    f"Daily Change: {data['change']:+.2f}<br>"
                    f"Change %: {change_pct:+.2f}%<br>"
                    f"RSI: {data['rsi']}<br>"
                    f"<extra></extra>"
                )
            ))

        # Update layout
        fig.update_layout(
            title="Market Indices Performance - Daily Change %",
            xaxis_title="",
            yaxis_title="Daily Change %",
            hovermode='closest',
            height=500,
            template='plotly_white',
            xaxis=dict(showgrid=False),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                gridwidth=1,
                gridcolor='lightgray'
            )
        )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

        # Add colored background zones (positive/negative)
        fig.add_hrect(
            y0=0, y1=max([d.get('change_percent', 0) for d in indices_data.values() if isinstance(d.get('change_percent'), (int, float))], default=10),
            fillcolor="green",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        fig.add_hrect(
            y0=min([d.get('change_percent', 0) for d in indices_data.values() if isinstance(d.get('change_percent'), (int, float))], default=-10),
            y1=0,
            fillcolor="red",
            opacity=0.05,
            layer="below",
            line_width=0
        )

        return fig
