"""
Data Fetcher Module
Fetches historical stock data from Polygon.io with SQLite caching
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import requests
import os
# from dotenv import load_dotenv
from database import CacheManager
import streamlit as st
# Load environment variables
# load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io API configuration
# POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', 'NJFsmgVJrgZxyLE0Rk0RUQvmXq66_IU_')
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
POLYGON_BASE_URL = "https://api.polygon.io/v2"


class DataFetcher:
    """Fetches historical stock price data with SQLite caching"""

    def __init__(self, symbol: str):
        """
        Initialize DataFetcher

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        """
        self.symbol = symbol.upper()
        self.data = None
        self.cache_manager = CacheManager()

    def fetch_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data from Polygon.io with SQLite caching

        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
            interval: Data interval (1d, 1wk, 1mo supported)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {self.symbol} data for {period} with {interval} interval...")

            # Step 1: Check cache first
            cached_data = self.cache_manager.get_cached_data(
                symbol=self.symbol,
                period=period,
                interval=interval
            )

            if cached_data is not None:
                logger.info(f"✓ Using cached data for {self.symbol}")
                self.data = cached_data
                return self.data

            # Step 2: Cache miss - fetch from API
            logger.info(f"Cache miss for {self.symbol}, fetching from API...")

            # Calculate date range
            end_date = datetime.now()
            period_mapping = {
                '1d': timedelta(days=1),
                '5d': timedelta(days=5),
                '1mo': timedelta(days=30),
                '3mo': timedelta(days=90),
                '6mo': timedelta(days=180),
                '1y': timedelta(days=365),
                '2y': timedelta(days=730),
                '5y': timedelta(days=1825),
                '10y': timedelta(days=3650),
            }
            start_date = end_date - period_mapping.get(period, timedelta(days=365))

            # Map interval to timespan
            timespan_mapping = {
                '1d': 'day',
                '1wk': 'week',
                '1mo': 'month'
            }
            timespan = timespan_mapping.get(interval, 'day')

            # Build API request
            url = f"{POLYGON_BASE_URL}/aggs/ticker/{self.symbol}/range/1/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apiKey': POLYGON_API_KEY,
                'sort': 'asc',
                'limit': 50000
            }

            # Retry logic
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()

                    data = response.json()

                    if 'results' not in data or not data['results']:
                        logger.error(f"No data found for symbol {self.symbol}")
                        return None

                    # Convert to DataFrame
                    records = []
                    for item in data['results']:
                        records.append({
                            'Date': pd.to_datetime(item['t'], unit='ms'),
                            'Open': item.get('o'),
                            'High': item.get('h'),
                            'Low': item.get('l'),
                            'Close': item.get('c'),
                            'Volume': item.get('v')
                        })

                    self.data = pd.DataFrame(records)
                    self.data.set_index('Date', inplace=True)
                    self.data.sort_index(inplace=True)

                    logger.info(f"Successfully fetched {len(self.data)} records")

                    # Step 3: Save to cache
                    try:
                        self.cache_manager.save_data(
                            symbol=self.symbol,
                            period=period,
                            interval=interval,
                            start_date=start_date,
                            end_date=end_date,
                            data=self.data
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save to cache: {str(e)}")
                        # Continue anyway - cache failure shouldn't block data return

                    return self.data

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {str(e)}")

            # Step 4: API failure - try stale cache as fallback
            try:
                stale_data = self.cache_manager.get_cached_data(
                    symbol=self.symbol,
                    period=period,
                    interval=interval,
                    allow_expired=True  # Allow expired cache as fallback
                )
                if stale_data is not None:
                    logger.warning(f"API error - using stale cache for {self.symbol}")
                    self.data = stale_data
                    return self.data
            except Exception as cache_error:
                logger.error(f"Could not retrieve stale cache: {str(cache_error)}")

            return None

    def get_latest_price(self) -> float:
        """Get the latest closing price"""
        if self.data is None or self.data.empty:
            return None
        return self.data['Close'].iloc[-1]

    def get_price_change(self) -> tuple:
        """
        Get price change over period

        Returns:
            Tuple of (change, percent_change) or (None, None) if no data
        """
        if self.data is None or self.data.empty:
            return None, None

        latest = self.data['Close'].iloc[-1]
        previous = self.data['Close'].iloc[0]
        change = latest - previous
        pct_change = (change / previous) * 100

        return change, pct_change

    def get_data(self) -> pd.DataFrame:
        """Return the current data"""
        return self.data

    def clear_cache(self, days: int = 7):
        """
        Clear old cache entries

        Args:
            days: Delete entries older than this many days
        """
        try:
            self.cache_manager.clear_old_cache(days=days)
            logger.info(f"Cleared cache entries older than {days} days")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def get_cache_stats(self):
        """Get cache statistics"""
        return self.cache_manager.get_cache_stats()


class MarketFetcher:
    """Fetches market index data for end-of-day analysis"""

    def __init__(self):
        """Initialize MarketFetcher with key market indices"""
        self.symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX']
        self.index_names = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq-100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility'
        }
        self.cache_manager = CacheManager()

    def fetch_market_indices(self) -> dict:
        """
        Fetch all 5 market indices with 1-day period (intraday data)

        Returns:
            Dictionary with structure:
            {
                'SPY': {
                    'symbol': 'SPY',
                    'name': 'S&P 500',
                    'price': 450.25,
                    'change': 2.50,
                    'change_percent': 0.56,
                    'rsi': 55.30,
                    'timestamp': '2024-02-07 15:30:00'
                },
                'QQQ': { ... },
                ...
            }
        """
        try:
            logger.info("Fetching market indices (SPY, QQQ, DIA, IWM, VIX)...")
            indices_data = {}

            for symbol in self.symbols:
                try:
                    # Fetch 1-day period data for each index
                    fetcher = DataFetcher(symbol)
                    data = fetcher.fetch_data(period='1d', interval='1d')

                    if data is not None and not data.empty:
                        # Extract metrics from the fetched data
                        latest_close = data['Close'].iloc[-1]
                        open_price = data['Open'].iloc[0]
                        daily_change = latest_close - open_price
                        daily_change_percent = (daily_change / open_price * 100) if open_price != 0 else 0

                        # Get RSI from analyzer
                        from analyzer import StockAnalyzer
                        analyzer = StockAnalyzer(symbol, period='1d')
                        analyzer.analyze()
                        latest_analysis = analyzer.get_latest_analysis()
                        rsi_value = latest_analysis.get('RSI') or 50  # Use 50 if RSI is None or not available

                        # Store the index data
                        indices_data[symbol] = {
                            'symbol': symbol,
                            'name': self.index_names[symbol],
                            'price': round(latest_close, 2),
                            'change': round(daily_change, 2),
                            'change_percent': round(daily_change_percent, 2),
                            'rsi': round(rsi_value, 2),
                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        logger.info(f"✓ {symbol}: ${latest_close:.2f} ({daily_change_percent:+.2f}%)")

                    else:
                        logger.warning(f"No data available for {symbol}")
                        indices_data[symbol] = {
                            'symbol': symbol,
                            'name': self.index_names[symbol],
                            'price': 'N/A',
                            'change': 'N/A',
                            'change_percent': 'N/A',
                            'rsi': 'N/A',
                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {str(e)}")
                    indices_data[symbol] = {
                        'symbol': symbol,
                        'name': self.index_names[symbol],
                        'price': 'N/A',
                        'change': 'N/A',
                        'change_percent': 'N/A',
                        'rsi': 'N/A',
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

            logger.info(f"Successfully fetched {len([x for x in indices_data.values() if x['price'] != 'N/A'])} market indices")
            return indices_data

        except Exception as e:
            logger.error(f"Error fetching market indices: {str(e)}")
            return None

    def fetch_news(self, limit=5):
        """
        Fetch latest news articles for the stock from Polygon.io

        Args:
            limit: Number of articles to fetch (default: 5)

        Returns:
            list: List of news articles with title, description, url, published_utc, author, image_url
        """
        try:
            logger.info(f"Fetching news for {self.symbol}")

            # Build news API request
            url = f"{self.base_url}/reference/news"
            params = {
                'ticker': self.symbol,
                'limit': limit,
                'sort': 'published_utc',
                'order': 'desc',
                'apiKey': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'OK' and 'results' in data:
                articles = []
                for article in data['results'][:limit]:
                    articles.append({
                        'title': article.get('title', 'No title'),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_utc': article.get('published_utc', ''),
                        'author': article.get('author', 'Unknown'),
                        'image_url': article.get('image_url', '')
                    })

                logger.info(f"Successfully fetched {len(articles)} articles for {self.symbol}")
                return articles
            else:
                logger.warning(f"No news found for {self.symbol}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news for {self.symbol}: {str(e)}")
            return []  # Return empty list on error

