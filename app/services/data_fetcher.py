# app/services/data_fetcher.py

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


def safe_float(value, default=None):
    """Safely convert value to float, returning None if conversion fails."""
    if value is None or pd.isna(value) or value == '':
        return default
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('%', '').strip()
        return float(value)
    except (ValueError, TypeError):
        return default


class DataFetcher:
    def __init__(self, newsapi_key: Optional[str] = None):
        self.screener_base_url = "https://www.screener.in"
        self.newsapi_key = newsapi_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            from app.services.news_analyzer import NewsSentimentAnalyzer
            self.news_analyzer = NewsSentimentAnalyzer(newsapi_key)
            logger.info("News analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize news analyzer: {e}")
            self.news_analyzer = self.FallbackAnalyzer()

    class FallbackAnalyzer:
        def analyze_news_sentiment(self, company_name, symbol): return {'news_sentiment': 0.5, 'articles': []}

        def get_company_news(self, company_name, symbol): return []

    def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Scrapes the main company page on Screener.in to extract fundamental data."""
        logger.info(f"Scraping fundamentals for {symbol} from Screener.in")
        page_url = f"{self.screener_base_url}/company/{symbol}/consolidated/"
        try:
            res = requests.get(page_url, headers=self.headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')

            fundamentals = {}
            top_ratios = soup.find('ul', id='top-ratios')
            if top_ratios:
                for li in top_ratios.find_all("li"):
                    name = li.find("span", class_="name").text.strip()
                    value = li.find("span", class_="number").text.strip()
                    if "Market Cap" in name:
                        fundamentals['market_cap'] = safe_float(value) * 10000000  # Convert Cr to absolute
                    elif "Current Price" in name:
                        fundamentals['current_price'] = safe_float(value)
                    elif "Stock P/E" in name:
                        fundamentals['pe_ratio'] = safe_float(value)
                    elif "Book Value" in name:
                        fundamentals['book_value'] = safe_float(value)
                    elif "Dividend Yield" in name:
                        fundamentals['dividend_yield'] = safe_float(value)
                    elif "ROCE" in name:
                        fundamentals['roce'] = safe_float(value)
                    elif "ROE" in name:
                        fundamentals['roe'] = safe_float(value)
            logger.info(f"Successfully scraped fundamentals for {symbol}.")
            return fundamentals
        except Exception as e:
            logger.error(f"Failed to scrape Screener.in for {symbol}: {e}", exc_info=True)
            return self._get_sample_fundamentals()

    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetches technical data using yfinance."""
        yf_symbol = f"{symbol}.NS"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        try:
            logger.info(f"Fetching technicals for {yf_symbol} using yfinance.")
            stock = yf.Ticker(yf_symbol)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty: raise ValueError(f"Yfinance returned empty dataframe for {symbol}")

            close_prices = hist['Close']
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            exp12 = close_prices.ewm(span=12, adjust=False).mean()
            exp26 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()

            return {
                "rsi": safe_float(rsi.iloc[-1], 50),
                "macd": safe_float(macd.iloc[-1], 0),
                "macd_signal": safe_float(signal.iloc[-1], 0),
                "current_price": safe_float(close_prices.iloc[-1]),
                "volume": safe_float(hist['Volume'].iloc[-1]),
                "52_week_high": safe_float(close_prices.max()),
                "52_week_low": safe_float(close_prices.min())
            }
        except Exception as e:
            logger.error(f"Yfinance failed for {symbol}: {e}", exc_info=True)
            return self._get_sample_technicals()

    def get_sentiment_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Fetches news sentiment data using the news analyzer."""
        return self.news_analyzer.analyze_news_sentiment(company_name, symbol)

    def _get_sample_fundamentals(self) -> Dict[str, Any]:
        return {}

    def _get_sample_technicals(self) -> Dict[str, Any]:
        return {}