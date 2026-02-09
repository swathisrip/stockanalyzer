# Stock Analysis Web App - Quick Start Guide

## What Was Created

I've converted your CLI stock analysis app into a **Streamlit web application** with:

### ✨ Features
1. **Single Stock Analysis** - Analyze any stock with full technical indicators
2. **Multi-Stock Comparison** - Compare multiple stocks side-by-side
3. **Interactive Charts** - Zoom, pan, hover tooltips with Plotly
4. **Trading Signals** - Visual indicators for BUY/SELL/BULLISH/BEARISH signals
5. **Export Data** - Download analysis and signals as CSV

### 📁 New Files Created
- `streamlit_app.py` - Main web app (entry point)
- `web_visualizer.py` - Plotly interactive charts
- `.streamlit/config.toml` - Streamlit configuration

### 🔄 Modified Files
- `requirements.txt` - Added streamlit, plotly, python-dotenv
- `data_fetcher.py` - Now reads API key from environment variables

### ✅ Unchanged (Still Work)
- `main.py` - CLI version still works
- `analyzer.py`, `indicators.py`, `config.py`, `visualizer.py`

---

## Installation

### 1. Install New Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (Optional - Already Configured)
The app uses your Polygon.io API key. You can optionally create a `.env` file:

```bash
# Create .env file in stockapp directory
echo POLYGON_API_KEY=NJFsmgVJrgZxyLE0Rk0RUQvmXq66_IU_ > .env
```

---

## Running the Web App

### Option 1: From Command Prompt/PowerShell
```bash
cd c:\Workspace\stockapp
streamlit run streamlit_app.py
```

### Option 2: From Any Directory
```bash
streamlit run c:\Workspace\stockapp\streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Using the Web App

### 📊 Tab 1: Single Stock Analysis
1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Choose time period (1d to 10y)
3. Select interval (daily, weekly, monthly)
4. Click "Analyze Stock"
5. View:
   - Current metrics (Price, RSI, MACD)
   - 4-panel technical indicator dashboard
   - Individual interactive charts
   - Trading signals timeline and table
   - Export buttons for CSV

### 📈 Tab 2: Multi-Stock Comparison
1. Enter multiple symbols separated by commas (e.g., AAPL,MSFT,GOOGL)
2. Choose time period
3. Click "Compare Stocks"
4. View:
   - Comparison table with all metrics
   - Side-by-side price charts
   - Side-by-side RSI charts
   - Side-by-side MACD charts

### ⚙️ Tab 3: Settings
- View API configuration
- See all indicator parameters
- Learn about trading signals
- Read disclaimer

---

## Interactive Features

### Charts Include
✅ **Zoom** - Click and drag to zoom into date ranges
✅ **Pan** - Move around the chart
✅ **Hover** - See exact values on hover
✅ **Legend Toggle** - Click legend items to show/hide lines
✅ **Download** - Camera icon to save chart as PNG

---

## Trading Signals Explained

| Signal | Condition | Color |
|--------|-----------|-------|
| 🔼 **BUY** | RSI < 30 + MACD > Signal | Green |
| 🔽 **SELL** | RSI > 70 + MACD < Signal | Red |
| 📈 **BULLISH** | MACD > Signal + RSI momentum (40-70) | Light Green |
| 📉 **BEARISH** | MACD < Signal + RSI decline (30-60) | Light Red |

---

## Troubleshooting

### Web app won't start
```bash
# Restart with verbose logging
streamlit run streamlit_app.py --logger.level=debug
```

### API errors
- Check internet connection
- Verify Polygon.io API key in data_fetcher.py
- Try again after a few seconds (rate limiting)

### Charts not showing
- Ensure plotly is installed: `pip install plotly`
- Try refreshing the browser

### Session state errors
- Clear browser cache
- Restart the app: Ctrl+C and rerun

---

## File Structure (Updated)

```
c:\Workspace\stockapp\
├── streamlit_app.py          ← Run this!
├── web_visualizer.py         ← Plotly charts
├── data_fetcher.py           ← Updated with env var
├── analyzer.py
├── indicators.py
├── config.py
├── main.py                   ← CLI version still works
├── requirements.txt          ← Updated
├── .streamlit/
│   └── config.toml           ← Streamlit config
├── .env                      ← (Optional) API key
└── README.md                 ← This file
```

---

## Tips

1. **Default**: App loads AAPL by default for 1 year of daily data
2. **Caching**: Data is cached for 5 minutes to avoid repeated API calls
3. **Performance**: Use shorter time periods (6 months) for faster loading
4. **CSV Export**: Download data for further analysis in Excel

---

## What's Next?

### Potential Enhancements
- [ ] Watchlist feature (save favorite stocks)
- [ ] Email alerts for trading signals
- [ ] Portfolio tracking
- [ ] Mobile-responsive design
- [ ] Dark mode theme
- [ ] More technical indicators
- [ ] Backtesting engine
- [ ] Database for historical signal tracking

---

## Disclaimer

This application is for **educational and informational purposes only**. It should not
be considered as financial advice. Always conduct your own research and consult with a
financial advisor before making investment decisions.

Technical indicators are tools to identify trends and potential entry/exit points, but
they are not foolproof. No single indicator should be relied upon for trading decisions.

---

## Support

For issues or questions:
1. Check the error messages in the terminal
2. Verify API key is set correctly
3. Try with a different stock symbol
4. Restart the application

Enjoy your stock analysis! 📊
