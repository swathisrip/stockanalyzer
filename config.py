# Configuration for Stock Analysis App

# Default parameters
DEFAULT_STOCK_SYMBOL = "AAPL"
DEFAULT_PERIOD = "1y"  # 1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max
DEFAULT_INTERVAL = "1d"  # 1 minute, 5 minutes, 15 minutes, 30 minutes, 60 minutes, 1 hour, 1 day, 1 week, 1 month, 3 months

# Technical Indicators Parameters
# RSI (Relative Strength Index)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD (Moving Average Convergence Divergence)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# EMA (Exponential Moving Average)
EMA_PERIODS = [20, 50, 200]

# SMA (Simple Moving Average)
SMA_PERIODS = [20, 50, 200]

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# Chart settings
CHART_FIGSIZE = (14, 10)
CHART_DPI = 100
CHART_STYLE = 'seaborn-v0_8-darkgrid'

# Signals
BUY_SIGNAL = "BUY"
SELL_SIGNAL = "SELL"
NEUTRAL_SIGNAL = "NEUTRAL"
