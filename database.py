"""
SQLite Cache Manager for Stock Data
Stores fetched stock data with 1-day expiration to reduce API calls
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages SQLite cache for stock data with 1-day expiration"""

    def __init__(self, db_path='stock_data.db'):
        """
        Initialize CacheManager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create database and table if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_period
                ON stock_cache(symbol, period, interval)
            ''')

            # Create market summary cache table for end-of-day analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_summary_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    indices_data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index for date lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_date
                ON market_summary_cache(date)
            ''')

            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def get_cached_data(self, symbol, period, interval, allow_expired=False):
        """
        Get cached data if valid (< 1 day old)

        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., '6mo', '1y')
            interval: Data interval (e.g., 'day', 'week')
            allow_expired: If True, return data even if > 1 day old

        Returns:
            DataFrame if valid cache exists, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query for matching record
            cursor.execute('''
                SELECT data, timestamp FROM stock_cache
                WHERE symbol = ? AND period = ? AND interval = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol.upper(), period, interval))

            result = cursor.fetchone()
            conn.close()

            if result is None:
                logger.debug(f"Cache miss for {symbol} {period} {interval}")
                return None

            data_json, timestamp_str = result

            # Check if cache is valid (< 1 day old)
            if not allow_expired:
                if not self.is_cache_valid(timestamp_str):
                    logger.info(f"Cache expired for {symbol} {period} {interval}")
                    return None

            # Deserialize JSON to DataFrame
            try:
                data_dict = json.loads(data_json)

                # Handle split-oriented format (with index, columns, data keys)
                if 'index' in data_dict and 'columns' in data_dict and 'data' in data_dict:
                    df = pd.DataFrame(
                        data=data_dict['data'],
                        index=data_dict['index'],
                        columns=data_dict['columns']
                    )
                    # Try to convert index to DatetimeIndex if it looks like dates
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        pass  # Index is not datetime, keep as is
                else:
                    # Fallback for old format (direct dict to DataFrame)
                    df = pd.DataFrame(data_dict)

                # Restore index name if it was preserved
                if 'index_name' in data_dict:
                    df.index.name = data_dict['index_name']

                logger.info(f"Cache hit for {symbol} {period} {interval} (timestamp: {timestamp_str})")
                return df

            except Exception as e:
                logger.error(f"Error deserializing cache data: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    def save_data(self, symbol, period, interval, start_date, end_date, data):
        """
        Save DataFrame to cache

        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., '6mo', '1y')
            interval: Data interval (e.g., 'day', 'week')
            start_date: Start date of data
            end_date: End date of data
            data: DataFrame to cache
        """
        try:
            # Serialize DataFrame to JSON using 'split' orientation
            # This puts index, columns, and data in separate keys instead of using index as dict keys
            data_dict = data.to_dict(orient='split')

            # Convert index values to strings to handle Timestamp objects
            if 'index' in data_dict and data_dict['index']:
                data_dict['index'] = [str(idx) for idx in data_dict['index']]

            # Preserve index name for restoration
            if data.index.name:
                data_dict['index_name'] = data.index.name

            data_json = json.dumps(data_dict)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete old entries for this symbol/period/interval
            cursor.execute('''
                DELETE FROM stock_cache
                WHERE symbol = ? AND period = ? AND interval = ?
            ''', (symbol.upper(), period, interval))

            # Insert new data
            cursor.execute('''
                INSERT INTO stock_cache
                (symbol, period, interval, start_date, end_date, data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol.upper(), period, interval, str(start_date), str(end_date), data_json))

            conn.commit()
            conn.close()

            logger.info(f"Cached {symbol} {period} {interval} ({len(data)} records)")

        except Exception as e:
            logger.error(f"Error saving cached data: {str(e)}")
            raise

    def is_cache_valid(self, timestamp_str):
        """
        Check if cache is < 1 day old

        Args:
            timestamp_str: Timestamp string from database (format: 'YYYY-MM-DD HH:MM:SS.sss')

        Returns:
            Boolean - True if cache is valid (< 1 day old)
        """
        try:
            # Parse timestamp
            cache_time = datetime.strptime(
                timestamp_str.split('.')[0],  # Remove milliseconds if present
                '%Y-%m-%d %H:%M:%S'
            )
            current_time = datetime.now()

            # Check if < 1 day old (86400 seconds)
            age_seconds = (current_time - cache_time).total_seconds()
            is_valid = age_seconds < 43200   # 1 day in seconds 86400

            if is_valid:
                hours_old = age_seconds / 3600
                logger.debug(f"Cache valid ({hours_old:.1f} hours old)")
            else:
                hours_old = age_seconds / 3600
                logger.debug(f"Cache expired ({hours_old:.1f} hours old)")

            return is_valid

        except Exception as e:
            logger.error(f"Error checking cache validity: {str(e)}")
            return False

    def clear_old_cache(self, days=7):
        """
        Delete cache entries older than specified days

        Args:
            days: Number of days to keep (default 7)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete records older than specified days
            cutoff_date = datetime.now() - timedelta(days=days)

            cursor.execute('''
                DELETE FROM stock_cache
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Cleaned {deleted_count} cache entries older than {days} days")

        except Exception as e:
            logger.error(f"Error clearing old cache: {str(e)}")
            raise

    def get_cache_stats(self):
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total entries
            cursor.execute('SELECT COUNT(*) FROM stock_cache')
            total_entries = cursor.fetchone()[0]

            # Get unique symbols
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM stock_cache')
            unique_symbols = cursor.fetchone()[0]

            # Get database size
            conn.close()
            db_size = os.path.getsize(self.db_path) / 1024  # Size in KB

            return {
                'total_entries': total_entries,
                'unique_symbols': unique_symbols,
                'database_size_kb': round(db_size, 2)
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return None

    def get_market_summary(self, allow_expired=False):
        """
        Get cached market summary if valid (< 1 day old)

        Args:
            allow_expired: If True, return data even if > 1 day old

        Returns:
            Dictionary with indices_data if valid cache exists, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query for today's market summary (most recent entry)
            cursor.execute('''
                SELECT indices_data, timestamp FROM market_summary_cache
                ORDER BY timestamp DESC
                LIMIT 1
            ''')

            result = cursor.fetchone()
            conn.close()

            if result is None:
                logger.info("No market summary cache found")
                return None

            indices_json, timestamp_str = result

            # Check if cache is valid (< 1 day old)
            if not allow_expired:
                if not self.is_market_summary_valid(timestamp_str):
                    logger.info("Market summary cache expired")
                    return None

            # Deserialize JSON to dictionary
            try:
                indices_data = json.loads(indices_json)
                logger.info(f"Market summary cache hit (timestamp: {timestamp_str})")
                return indices_data
            except Exception as e:
                logger.error(f"Error deserializing market summary: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving market summary: {str(e)}")
            return None

    def save_market_summary(self, indices_data):
        """
        Save market summary snapshot to cache

        Args:
            indices_data: Dictionary with market indices data (must be JSON-serializable)
        """
        try:
            # Get today's date for the date field
            today = datetime.now().strftime('%Y-%m-%d')

            # Serialize dictionary to JSON
            indices_json = json.dumps(indices_data)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete old entries for today (keep only latest)
            cursor.execute('''
                DELETE FROM market_summary_cache
                WHERE date = ?
            ''', (today,))

            # Insert new market summary
            cursor.execute('''
                INSERT INTO market_summary_cache
                (date, indices_data, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (today, indices_json))

            conn.commit()
            conn.close()

            logger.info(f"Saved market summary cache for {today}")

        except Exception as e:
            logger.error(f"Error saving market summary: {str(e)}")
            raise

    def is_market_summary_valid(self, timestamp_str):
        """
        Check if market summary cache is < 1 day old

        Args:
            timestamp_str: Timestamp string from database

        Returns:
            Boolean - True if cache is valid (< 1 day old)
        """
        try:
            # Parse timestamp
            cache_time = datetime.strptime(
                timestamp_str.split('.')[0],
                '%Y-%m-%d %H:%M:%S'
            )
            current_time = datetime.now()

            # Check if < 1 day old (86400 seconds)
            age_seconds = (current_time - cache_time).total_seconds()
            is_valid = age_seconds < 86400

            if is_valid:
                hours_old = age_seconds / 3600
                logger.debug(f"Market summary valid ({hours_old:.1f} hours old)")
            else:
                hours_old = age_seconds / 3600
                logger.debug(f"Market summary expired ({hours_old:.1f} hours old)")

            return is_valid

        except Exception as e:
            logger.error(f"Error checking market summary validity: {str(e)}")
            return False
