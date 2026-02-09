"""
Stock Analysis Web App
Built with Streamlit and Plotly for interactive technical analysis
"""

import streamlit as st
import pandas as pd
from analyzer import StockAnalyzer
from web_visualizer import WebVisualizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Stock Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .signal-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .buy-signal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .sell-signal {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .bullish-signal {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .bearish-signal {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Technical Analysis Dashboard")
st.markdown("Real-time technical analysis with RSI, MACD, and trading signals")

# ==================== HELPER FUNCTIONS FOR MULTI-PERIOD VIEW ====================
def display_multi_period_view(multi_analyzers, symbol):
    """Display 3x3 grid: 3 intervals × 3 chart types"""
    st.subheader("Multi-Period Technical Analysis")

    tab_full, tab_zoom = st.tabs(["📊 Full Data View", "🔍 Recent Period (Last 60 Candles)"])

    with tab_full:
        display_period_grid(multi_analyzers, symbol, zoom=False)

    with tab_zoom:
        display_period_grid(multi_analyzers, symbol, zoom=True)

def display_period_grid(multi_analyzers, symbol, zoom=False):
    """Display charts in 3 columns (Daily, Weekly, Monthly)"""
    interval_labels = {'1d': 'Daily', '1wk': 'Weekly', '1mo': 'Monthly'}
    cols = st.columns(3, gap="medium")

    for col_idx, (interval, analyzer) in enumerate(multi_analyzers.items()):
        with cols[col_idx]:
            st.markdown(f"#### {interval_labels[interval]}")

            # Skip if no data
            if analyzer.data is None or len(analyzer.data) == 0:
                st.warning(f"No data available for {interval_labels[interval]}")
                continue

            visualizer = WebVisualizer(analyzer.data, symbol)

            # Calculate zoom range (last 60 candles)
            zoom_start = None
            zoom_end = None
            if zoom and len(analyzer.data) > 60:
                zoom_start = analyzer.data.index[-60]
                zoom_end = analyzer.data.index[-1]

            # Price chart
            try:
                fig_price = visualizer.plot_price_with_moving_averages()
                if zoom and zoom_start is not None and zoom_end is not None:
                    fig_price.update_xaxes(range=[zoom_start, zoom_end])
                st.plotly_chart(fig_price, use_container_width=True, key=f"p_{interval}_{zoom}")
            except Exception as e:
                st.error(f"Error rendering price chart: {str(e)}")

            # RSI chart
            try:
                fig_rsi = visualizer.plot_rsi()
                if zoom and zoom_start is not None and zoom_end is not None:
                    fig_rsi.update_xaxes(range=[zoom_start, zoom_end])
                st.plotly_chart(fig_rsi, use_container_width=True, key=f"r_{interval}_{zoom}")
            except Exception as e:
                st.error(f"Error rendering RSI chart: {str(e)}")

            # MACD chart
            try:
                fig_macd = visualizer.plot_macd()
                if zoom and zoom_start is not None and zoom_end is not None:
                    fig_macd.update_xaxes(range=[zoom_start, zoom_end])
                st.plotly_chart(fig_macd, use_container_width=True, key=f"m_{interval}_{zoom}")
            except Exception as e:
                st.error(f"Error rendering MACD chart: {str(e)}")

def display_stock_news(news_articles):
    """Display stock news in card format"""
    if not news_articles:
        st.info("📰 No recent news available for this stock")
        return

    st.subheader("📰 Latest News")

    for idx, article in enumerate(news_articles):
        with st.container():
            # Create columns for content and image
            has_image = article.get('image_url') and article['image_url'].strip()
            if has_image:
                col1, col2 = st.columns([3, 1])
            else:
                col1 = st.columns(1)[0]
                col2 = None

            with col1:
                # Headline
                st.markdown(f"**{article.get('title', 'No title')}**")

                # Description
                description = article.get('description', '')
                if description:
                    truncated = (description[:200] + "...") if len(description) > 200 else description
                    st.markdown(f"*{truncated}*")

                # Metadata row
                col_date, col_source = st.columns(2)

                with col_date:
                    # Parse and format date
                    try:
                        pub_date = pd.to_datetime(article.get('published_utc', ''))
                        st.caption(f"📅 {pub_date.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"📅 {article.get('published_utc', '')}")

                with col_source:
                    author = article.get('author', 'Unknown Source')
                    st.caption(f"📝 {author}")

                # Read more link
                if article.get('url'):
                    st.markdown(f"[🔗 Read Full Article]({article['url']})")

            if col2 and has_image:
                with col2:
                    try:
                        st.image(article['image_url'], width=150)
                    except:
                        pass  # Skip image if loading fails

            st.divider()

# ==================== CREATE TABS ====================
# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Single Stock Analysis", "📈 Multi-Stock Comparison", "📋 Watchlist", "🌍 Market Summary", "⚙️ Settings", "📊 Multi-Period Analysis"])

# ==================== TAB 1: SINGLE STOCK ANALYSIS ====================
with tab1:
    st.header("Single Stock Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()

    with col2:
        period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=5  # Default to 1y
        )

    with col3:
        interval = st.selectbox(
            "Interval",
            ["1d", "1wk", "1mo"],
            index=0  # Default to 1d
        )

    # Analyze button
    analyze_button = st.button("🔍 Analyze Stock", key="analyze_single", use_container_width=True)

    if analyze_button or st.session_state.get('analyzed_symbol') == symbol:
        st.session_state['analyzed_symbol'] = symbol

        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Create analyzer and run analysis
                analyzer = StockAnalyzer(symbol, period=period, interval=interval)

                if analyzer.analyze():
                    # Display metrics in columns
                    st.success(f"✅ Successfully loaded {symbol}")

                    # Get latest analysis
                    analysis = analyzer.get_latest_analysis()

                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Close Price", f"${analysis['Close Price']:.2f}")

                    with col2:
                        rsi_value = analysis['RSI']
                        rsi_color = "🟢" if rsi_value < 30 else "🔴" if rsi_value > 70 else "🟡"
                        st.metric("RSI (14)", f"{rsi_value:.2f}", delta=rsi_color)

                    with col3:
                        macd_value = analysis['MACD']
                        macd_signal = analysis['MACD Signal']
                        macd_diff = macd_value - macd_signal
                        st.metric("MACD", f"{macd_value:.4f}", delta=f"{macd_diff:+.4f}")

                    with col4:
                        # Price change
                        change, pct_change = analyzer.data_fetcher.get_price_change()
                        if change is not None:
                            st.metric("Period Change", f"${change:.2f}", delta=f"{pct_change:.2f}%")

                    # Fetch and display news
                    st.markdown("---")
                    try:
                        news_articles = analyzer.data_fetcher.fetch_news(limit=5)
                        if news_articles:
                            display_stock_news(news_articles)
                        else:
                            st.info("📰 No recent news available for this stock")
                    except Exception as e:
                        logger.warning(f"Could not fetch news: {str(e)}")
                        # Silently fail - don't show error, just skip news

                    # Current Signal Status
                    st.markdown("---")

                    latest_signal_row = analyzer.signals.iloc[-1]
                    signal_text = ""
                    signal_type = ""

                    if latest_signal_row['Buy_Signal'] == 1:
                        signal_text = "🔼 **BUY SIGNAL DETECTED**"
                        signal_type = "buy"
                        st.markdown(f'<div class="signal-box buy-signal">{signal_text}<br><small>RSI Oversold + MACD Bullish</small></div>', unsafe_allow_html=True)
                    elif latest_signal_row['Sell_Signal'] == 1:
                        signal_text = "🔽 **SELL SIGNAL DETECTED**"
                        signal_type = "sell"
                        st.markdown(f'<div class="signal-box sell-signal">{signal_text}<br><small>RSI Overbought + MACD Bearish</small></div>', unsafe_allow_html=True)
                    elif latest_signal_row['Bullish_Signal'] == 1:
                        signal_text = "📈 **BULLISH SIGNAL**"
                        signal_type = "bullish"
                        st.markdown(f'<div class="signal-box bullish-signal">{signal_text}<br><small>MACD Bullish + RSI Momentum</small></div>', unsafe_allow_html=True)
                    elif latest_signal_row['Bearish_Signal'] == 1:
                        signal_text = "📉 **BEARISH SIGNAL**"
                        signal_type = "bearish"
                        st.markdown(f'<div class="signal-box bearish-signal">{signal_text}<br><small>MACD Bearish + RSI Decline</small></div>', unsafe_allow_html=True)
                    else:
                        st.info("⚪ No active signals - Market conditions neutral")

                    # Display charts
                    st.markdown("---")
                    st.subheader("Technical Indicator Charts")

                    visualizer = WebVisualizer(analyzer.data, symbol)

                    # All indicators dashboard
                    st.plotly_chart(visualizer.plot_all_indicators(), use_container_width=True)

                    # Support/Resistance Levels
                    st.markdown("---")
                    st.subheader("Support/Resistance Levels (Pivot Points)")

                    pivot_levels = analyzer.indicators.calculate_pivot_points()
                    current_price = analyzer.data['Close'].iloc[-1]
                    sr_status = analyzer.indicators.get_sr_status(current_price, pivot_levels)

                    if pivot_levels:
                        # Display S/R chart
                        st.plotly_chart(visualizer.plot_price_with_sr_levels(pivot_levels), use_container_width=True)

                        # Display S/R levels table
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Pivot Point Levels:**")
                            sr_table_data = {
                                'Level': ['Strong Resistance (R2)', 'Resistance (R1)', 'Pivot Point', 'Support (S1)', 'Strong Support (S2)'],
                                'Price': [
                                    f"${pivot_levels['R2']:.2f}",
                                    f"${pivot_levels['R1']:.2f}",
                                    f"${pivot_levels['Pivot']:.2f}",
                                    f"${pivot_levels['S1']:.2f}",
                                    f"${pivot_levels['S2']:.2f}"
                                ],
                                'Distance': [
                                    f"${abs(current_price - pivot_levels['R2']):.2f} away",
                                    f"${abs(current_price - pivot_levels['R1']):.2f} away",
                                    f"${abs(current_price - pivot_levels['Pivot']):.2f} away",
                                    f"${abs(current_price - pivot_levels['S1']):.2f} away",
                                    f"${abs(current_price - pivot_levels['S2']):.2f} away"
                                ]
                            }
                            sr_df = pd.DataFrame(sr_table_data)
                            st.dataframe(sr_df, use_container_width=True, hide_index=True)

                        with col2:
                            st.write("**Current Market Position:**")
                            st.metric("Current Price", f"${current_price:.2f}")
                            st.metric("Zone", sr_status.get('zone', 'Unknown'))

                            if sr_status.get('nearest_resistance'):
                                res_name, res_price = sr_status['nearest_resistance']
                                st.metric("Nearest Resistance", f"{res_name} @ ${res_price:.2f}")

                            if sr_status.get('nearest_support'):
                                sup_name, sup_price = sr_status['nearest_support']
                                st.metric("Nearest Support", f"{sup_name} @ ${sup_price:.2f}")

                    # Trading signals timeline
                    st.markdown("---")
                    st.subheader("Trading Signals Timeline")
                    recent_signals = analyzer.get_recent_signals(days=len(analyzer.data))
                    if len(recent_signals) > 0:
                        st.plotly_chart(visualizer.plot_signals_timeline(recent_signals), use_container_width=True)

                        # Display signals table
                        st.subheader("Recent Trading Signals")
                        signals_display = recent_signals[['Date', 'Close', 'Signal', 'Type']].copy()
                        signals_display['Date'] = signals_display['Date'].dt.strftime('%Y-%m-%d')
                        signals_display['Close'] = signals_display['Close'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(signals_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No trading signals detected in this period")

                    # Export section
                    st.markdown("---")
                    st.subheader("Export Data")

                    col1, col2 = st.columns(2)

                    with col1:
                        # CSV export button
                        csv = analyzer.data.to_csv(index=True)
                        st.download_button(
                            label="📥 Download Analysis (CSV)",
                            data=csv,
                            file_name=f"{symbol}_analysis.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Signals export
                        if len(recent_signals) > 0:
                            signals_csv = recent_signals.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Signals (CSV)",
                                data=signals_csv,
                                file_name=f"{symbol}_signals.csv",
                                mime="text/csv"
                            )

                else:
                    st.error(f"Failed to analyze {symbol}. Please check the symbol and try again.")

            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                logger.error(f"Error in analysis: {str(e)}")

# ==================== TAB 2: MULTI-STOCK COMPARISON ====================
with tab2:
    st.header("Multi-Stock Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbols_input = st.text_input(
            "Stock Symbols",
            value="AAPL,MSFT,GOOGL",
            placeholder="Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL)"
        )

    with col2:
        period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,  # Default to 3mo
            key="comparison_period"
        )

    compare_button = st.button("🔍 Compare Stocks", key="analyze_multi", use_container_width=True)

    if compare_button or st.session_state.get('compared_symbols') == symbols_input:
        st.session_state['compared_symbols'] = symbols_input

        with st.spinner("Analyzing stocks..."):
            try:
                symbols = [s.strip().upper() for s in symbols_input.split(",")]

                comparison_data = []
                analyzers = {}

                for symbol in symbols:
                    try:
                        analyzer = StockAnalyzer(symbol, period=period)
                        if analyzer.analyze():
                            analyzers[symbol] = analyzer
                            analysis = analyzer.get_latest_analysis()
                            change, pct_change = analyzer.data_fetcher.get_price_change()

                            comparison_data.append({
                                'Symbol': symbol,
                                'Price': f"${analysis['Close Price']:.2f}",
                                'RSI': f"{analysis['RSI']:.2f}",
                                'MACD': f"{analysis['MACD']:.4f}",
                                'Signal': f"{analysis['MACD Signal']:.4f}",
                                'Change': f"{change:+.2f}" if change else "N/A",
                                'Change %': f"{pct_change:+.2f}%" if pct_change else "N/A"
                            })
                    except Exception as e:
                        st.warning(f"Could not analyze {symbol}: {str(e)}")
                        logger.error(f"Error analyzing {symbol}: {str(e)}")

                if comparison_data:
                    st.success(f"✅ Successfully analyzed {len(comparison_data)} stocks")

                    # Display comparison table
                    st.subheader("Stock Comparison Table")
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

                    # Side-by-side charts
                    if len(analyzers) > 1:
                        st.markdown("---")
                        st.subheader("Price Charts Comparison")

                        cols = st.columns(len(analyzers))
                        for idx, (symbol, analyzer) in enumerate(analyzers.items()):
                            with cols[idx]:
                                visualizer = WebVisualizer(analyzer.data, symbol)
                                st.plotly_chart(visualizer.plot_price_with_moving_averages(), use_container_width=True)

                        st.markdown("---")
                        st.subheader("RSI Comparison")

                        cols = st.columns(len(analyzers))
                        for idx, (symbol, analyzer) in enumerate(analyzers.items()):
                            with cols[idx]:
                                visualizer = WebVisualizer(analyzer.data, symbol)
                                st.plotly_chart(visualizer.plot_rsi(), use_container_width=True)

                        st.markdown("---")
                        st.subheader("MACD Comparison")

                        cols = st.columns(len(analyzers))
                        for idx, (symbol, analyzer) in enumerate(analyzers.items()):
                            with cols[idx]:
                                visualizer = WebVisualizer(analyzer.data, symbol)
                                st.plotly_chart(visualizer.plot_macd(), use_container_width=True)

                else:
                    st.error("Failed to analyze any stocks. Please check the symbols and try again.")

            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")
                logger.error(f"Error: {str(e)}")

# ==================== TAB 3: WATCHLIST ====================
with tab3:
    st.header("Watchlist Analysis")
    st.markdown("Upload a CSV file with stock symbols to analyze them all at once")

    col1, col2 = st.columns([2, 1])

    with col1:
        # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with stock symbols",
            type="csv",
            help="CSV should have 'Symbol' column or just one symbol per row"
        )

    with col2:
        # Period selector
        watchlist_period = st.selectbox(
            "Analysis Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=4,  # Default to 6mo
            key="watchlist_period"
        )

    analyze_watchlist = st.button("📊 Analyze Watchlist", key="analyze_watchlist", use_container_width=True)

    if uploaded_file and (analyze_watchlist or st.session_state.get('watchlist_analyzed')):
        st.session_state['watchlist_analyzed'] = True

        with st.spinner("Analyzing watchlist..."):
            try:
                # Read CSV
                df_watchlist = pd.read_csv(uploaded_file)

                # Extract symbols - handle both 'Symbol' column and simple list
                if 'Symbol' in df_watchlist.columns:
                    symbols = df_watchlist['Symbol'].str.upper().str.strip().tolist()
                elif 'symbol' in df_watchlist.columns:
                    symbols = df_watchlist['symbol'].str.upper().str.strip().tolist()
                else:
                    # Assume first column is symbols
                    symbols = df_watchlist.iloc[:, 0].astype(str).str.upper().str.strip().tolist()

                # Remove duplicates while preserving order
                symbols = list(dict.fromkeys(symbols))

                if not symbols:
                    st.error("No valid stock symbols found in CSV")
                else:
                    st.success(f"📋 Found {len(symbols)} stocks to analyze")

                    # Analyze each symbol
                    watchlist_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, symbol in enumerate(symbols):
                        status_text.text(f"Analyzing {idx + 1}/{len(symbols)}: {symbol}...")

                        try:
                            # Create analyzer and run analysis
                            analyzer = StockAnalyzer(symbol, period=watchlist_period)

                            if analyzer.analyze():
                                analysis = analyzer.get_latest_analysis()
                                latest_signal_row = analyzer.signals.iloc[-1]

                                # Determine signal
                                signal_text = "No Signal"
                                signal_type = "none"

                                if latest_signal_row['Buy_Signal'] == 1:
                                    signal_text = "BUY 🔼"
                                    signal_type = "BUY"
                                elif latest_signal_row['Sell_Signal'] == 1:
                                    signal_text = "SELL 🔽"
                                    signal_type = "SELL"
                                elif latest_signal_row['Bullish_Signal'] == 1:
                                    signal_text = "BULLISH 📈"
                                    signal_type = "BULLISH"
                                elif latest_signal_row['Bearish_Signal'] == 1:
                                    signal_text = "BEARISH 📉"
                                    signal_type = "BEARISH"

                                watchlist_results.append({
                                    'Symbol': symbol,
                                    'Price': analysis['Close Price'],
                                    'RSI': analysis['RSI'],
                                    'MACD': analysis['MACD'],
                                    'Signal': signal_text,
                                    'Type': signal_type
                                })

                        except Exception as e:
                            logger.warning(f"Could not analyze {symbol}: {str(e)}")
                            watchlist_results.append({
                                'Symbol': symbol,
                                'Price': 'N/A',
                                'RSI': 'N/A',
                                'MACD': 'N/A',
                                'Signal': 'ERROR',
                                'Type': 'error'
                            })

                        progress_bar.progress((idx + 1) / len(symbols))

                    status_text.text(f"✅ Analysis complete!")

                    # Display results
                    if watchlist_results:
                        # Format results for display
                        display_results = []
                        for result in watchlist_results:
                            if result['Type'] != 'error':
                                display_results.append({
                                    'Symbol': result['Symbol'],
                                    'Price': f"${result['Price']:.2f}",
                                    'RSI': f"{result['RSI']:.2f}",
                                    'MACD': f"{result['MACD']:.4f}",
                                    'Signal': result['Signal']
                                })
                            else:
                                display_results.append({
                                    'Symbol': result['Symbol'],
                                    'Price': 'ERROR',
                                    'RSI': 'ERROR',
                                    'MACD': 'ERROR',
                                    'Signal': 'ERROR'
                                })

                        st.markdown("---")
                        st.subheader("Watchlist Results")
                        df_results = pd.DataFrame(display_results)
                        st.dataframe(df_results, use_container_width=True, hide_index=True)

                        # Summary statistics
                        st.markdown("---")
                        st.subheader("Summary Statistics")

                        # Count signals
                        buy_count = sum(1 for r in watchlist_results if r['Type'] == 'BUY')
                        sell_count = sum(1 for r in watchlist_results if r['Type'] == 'SELL')
                        bullish_count = sum(1 for r in watchlist_results if r['Type'] == 'BULLISH')
                        bearish_count = sum(1 for r in watchlist_results if r['Type'] == 'BEARISH')
                        neutral_count = sum(1 for r in watchlist_results if r['Type'] == 'none')
                        error_count = sum(1 for r in watchlist_results if r['Type'] == 'error')

                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        with col1:
                            st.metric("Total Stocks", len([r for r in watchlist_results if r['Type'] != 'error']))

                        with col2:
                            st.metric("🔼 BUY Signals", buy_count)

                        with col3:
                            st.metric("🔽 SELL Signals", sell_count)

                        with col4:
                            st.metric("📈 BULLISH", bullish_count)

                        with col5:
                            st.metric("📉 BEARISH", bearish_count)

                        with col6:
                            st.metric("⚪ No Signal", neutral_count)

                        if error_count > 0:
                            st.warning(f"⚠️ Failed to analyze {error_count} stocks")

                        # Export button
                        st.markdown("---")
                        csv_export = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv_export,
                            file_name="watchlist_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Error processing watchlist: {str(e)}")
                logger.error(f"Watchlist error: {str(e)}")

# ==================== TAB 4: MARKET SUMMARY ====================
with tab4:
    st.header("🌍 Market Summary - End of Day Analysis")
    st.markdown("Overview of major market indices for daily trend analysis")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Refresh Market Data", use_container_width=True):
            st.session_state['refresh_market'] = True

    if st.session_state.get('refresh_market') or True:  # Always show fresh data initially
        with st.spinner("Fetching market indices..."):
            try:
                from data_fetcher import MarketFetcher
                from database import CacheManager

                market_fetcher = MarketFetcher()
                cache_manager = CacheManager()

                # Check if user explicitly clicked refresh button
                if st.session_state.get('refresh_market'):
                    # Bypass cache and fetch fresh data from API
                    indices_data = market_fetcher.fetch_market_indices()

                    if indices_data:
                        # Save to cache
                        try:
                            cache_manager.save_market_summary(indices_data)
                            st.success("✅ Market data refreshed from API and cached for 24 hours")
                        except Exception as e:
                            st.warning(f"Could not cache market data: {str(e)}")

                    st.session_state['refresh_market'] = False
                else:
                    # Normal load - try cache first, then fallback to API
                    cached_summary = cache_manager.get_market_summary(allow_expired=False)

                    if cached_summary is not None:
                        st.info("📊 Using cached market summary (less than 24 hours old)")
                        indices_data = cached_summary
                    else:
                        # Cache miss - fetch fresh data
                        indices_data = market_fetcher.fetch_market_indices()

                        if indices_data:
                            # Save to cache
                            try:
                                cache_manager.save_market_summary(indices_data)
                                st.success("✅ Market data cached for 24 hours")
                            except Exception as e:
                                st.warning(f"Could not cache market data: {str(e)}")

                if indices_data:
                    st.session_state['refresh_market'] = False

                    # Display 5 metric cards
                    st.markdown("---")
                    st.subheader("Market Indices Snapshot")

                    cols = st.columns(5)
                    valid_indices = {}

                    for idx, (symbol, col) in enumerate(zip(['SPY', 'QQQ', 'DIA', 'IWM', 'VIX'], cols)):
                        data = indices_data.get(symbol, {})

                        with col:
                            st.metric(
                                label=f"{symbol}",
                                value=data.get('price', 'N/A'),
                                delta=f"{data.get('change_percent', 0):+.2f}%" if isinstance(data.get('change_percent'), (int, float)) else 'N/A'
                            )

                            # Display RSI if available
                            if isinstance(data.get('rsi'), (int, float)):
                                rsi_value = data['rsi']
                                rsi_color = "🟢" if rsi_value < 30 else "🔴" if rsi_value > 70 else "🟡"
                                st.caption(f"RSI: {rsi_value:.1f} {rsi_color}")

                            # Store for sentiment calculation
                            if isinstance(data.get('change_percent'), (int, float)):
                                valid_indices[symbol] = data

                    # Calculate market sentiment
                    st.markdown("---")
                    st.subheader("Market Sentiment")

                    if valid_indices:
                        # Count positive indices (excluding VIX which is inverse)
                        positive_count = sum(1 for sym, data in valid_indices.items()
                                           if sym != 'VIX' and data.get('change_percent', 0) > 0)
                        bullish_count = len([s for s in ['SPY', 'QQQ', 'DIA', 'IWM'] if s in valid_indices])

                        # Sentiment logic
                        if positive_count >= 3:
                            sentiment = "BULLISH 📈"
                            sentiment_color = "green"
                            explanation = f"{positive_count}/4 indices trending up"
                        elif positive_count == 2:
                            sentiment = "NEUTRAL 🔷"
                            sentiment_color = "blue"
                            explanation = "Mixed market signals"
                        else:
                            sentiment = "BEARISH 📉"
                            sentiment_color = "red"
                            explanation = f"Market weakness - only {positive_count}/4 indices positive"

                        # Check VIX for additional confirmation
                        vix_data = valid_indices.get('VIX', {})
                        if isinstance(vix_data.get('change_percent'), (int, float)):
                            vix_change = vix_data['change_percent']
                            if vix_change > 5 and sentiment != "BEARISH 📉":
                                sentiment = "CAUTIOUS ⚠️"
                                sentiment_color = "orange"
                                explanation += " (VIX spike detected)"

                        col1, col2 = st.columns([2, 2])

                        with col1:
                            st.markdown(f"""
                            <div style='
                                background-color: {sentiment_color}20;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 4px solid {sentiment_color};
                                text-align: center;
                            '>
                                <h2 style='margin: 0; color: {sentiment_color};'>{sentiment}</h2>
                                <p style='margin: 10px 0 0 0; color: gray;'>{explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            # Display last update time
                            if indices_data:
                                first_timestamp = list(indices_data.values())[0].get('timestamp', 'Unknown')
                                st.info(f"📅 Last Updated\n\n{first_timestamp}")

                    # Chart: Intraday Performance
                    st.markdown("---")
                    st.subheader("Daily Performance Chart")

                    try:
                        from web_visualizer import WebVisualizer
                        chart = WebVisualizer.plot_market_indices(indices_data)
                        st.plotly_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render market chart: {str(e)}")

                else:
                    st.error("Failed to fetch market indices. Please try again.")

            except Exception as e:
                st.error(f"❌ Error fetching market summary: {str(e)}")
                logger.error(f"Market summary error: {str(e)}")

# ==================== TAB 5: SETTINGS ====================
with tab5:
    st.header("Settings & Configuration")

    st.subheader("API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📌 **API Status**")
        st.write("**Service**: Polygon.io")
        st.write("**Status**: ✅ Active")
        st.write("**API Key**: Configured from environment")

    with col2:
        st.info("📊 **Data Source**")
        st.write("- Real-time stock data from Polygon.io")
        st.write("- Daily, weekly, monthly intervals supported")
        st.write("- Up to 10 years of historical data available")

    st.markdown("---")

    st.subheader("Indicator Parameters (Read-Only)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**RSI Settings**")
        st.write("- Period: 14")
        st.write("- Overbought: 70")
        st.write("- Oversold: 30")

    with col2:
        st.write("**MACD Settings**")
        st.write("- Fast EMA: 12")
        st.write("- Slow EMA: 26")
        st.write("- Signal Line: 9")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Moving Averages**")
        st.write("- EMA Periods: 20, 50, 200")
        st.write("- SMA Periods: 20, 50, 200")

    with col2:
        st.write("**Bollinger Bands**")
        st.write("- Period: 20")
        st.write("- Standard Deviations: 2")

    st.markdown("---")

    st.subheader("About This App")
    st.write("""
    ### Stock Technical Analysis Dashboard

    This web application provides real-time technical analysis for stocks using:

    - **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
    - **MACD (Moving Average Convergence Divergence)**: Detects trend changes
    - **Moving Averages**: Confirms long-term trends
    - **Bollinger Bands**: Identifies volatility and support/resistance levels

    #### Trading Signal Criteria

    - **BUY Signal**: RSI < 30 (oversold) + MACD > Signal Line (bullish)
    - **SELL Signal**: RSI > 70 (overbought) + MACD < Signal Line (bearish)
    - **BULLISH Signal**: MACD > Signal Line + RSI > 40 + RSI ≤ 70
    - **BEARISH Signal**: MACD < Signal Line + RSI < 60 + RSI ≥ 30

    #### Data Source

    Stock data is sourced from **Polygon.io** API with real-time updates.

    #### Disclaimer

    This tool is for educational and informational purposes only. It should not be
    considered as financial advice. Always conduct your own research and consult with
    a financial advisor before making investment decisions.
    """)

# ==================== TAB 6: MULTI-PERIOD ANALYSIS ====================
with tab6:
    st.header("Multi-Period Technical Analysis")
    st.markdown("Compare Daily, Weekly, and Monthly trends side-by-side")

    col1, col2 = st.columns(2)

    with col1:
        mp_symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)",
            key="mp_symbol"
        ).upper()

    with col2:
        mp_period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=5,  # Default to 1y
            key="mp_period"
        )

    # Analyze button for multi-period
    mp_analyze_button = st.button("🔍 Analyze Multiple Periods", key="analyze_multi_period", use_container_width=True)

    if mp_analyze_button or st.session_state.get('mp_analyzed_symbol') == mp_symbol:
        st.session_state['mp_analyzed_symbol'] = mp_symbol

        with st.spinner(f"Analyzing {mp_symbol} across multiple periods..."):
            try:
                # Fetch data for all 3 intervals
                mp_analyzers = {}
                st.info("📊 Fetching data for Daily, Weekly, and Monthly intervals...")

                for mp_interval in ['1d', '1wk', '1mo']:
                    mp_analyzer = StockAnalyzer(mp_symbol, period=mp_period, interval=mp_interval)
                    if mp_analyzer.analyze():
                        mp_analyzers[mp_interval] = mp_analyzer
                    else:
                        st.warning(f"Could not analyze {mp_symbol} for {mp_interval} interval")

                if len(mp_analyzers) == 3:
                    st.success("✅ Multi-period data loaded successfully")
                    st.markdown("---")

                    # Display multi-period view
                    display_multi_period_view(mp_analyzers, mp_symbol)
                else:
                    st.error(f"Failed to fetch all periods for {mp_symbol}. Please try again.")

            except Exception as e:
                st.error(f"❌ Error analyzing {mp_symbol}: {str(e)}")
                logger.error(f"Multi-period analysis error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Stock Technical Analysis Dashboard v1.0 | Built with Streamlit & Plotly</p>
    <p>Data provided by Polygon.io | For educational and informational purposes only</p>
    </div>
""", unsafe_allow_html=True)
