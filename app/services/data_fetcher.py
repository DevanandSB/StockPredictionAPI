import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import random

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
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            from app.services.news_analyzer import NewsSentimentAnalyzer
            self.news_analyzer = NewsSentimentAnalyzer(newsapi_key)
        except Exception as e:
            logger.error(f"Failed to initialize news analyzer: {e}")
            self.news_analyzer = self.FallbackAnalyzer()

    class FallbackAnalyzer:
        def analyze_news_sentiment(self, company_name, symbol):
            return {'news_sentiment': 0.5, 'articles': [], 'social_media_sentiment': 0.5, 'analyst_rating': 7.5}

    def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Scrapes fundamental data from Screener.in with a fallback to yfinance for missing critical data.
        """
        logger.info(f"Scraping fundamentals for {symbol} from Screener.in")
        page_url = f"{self.screener_base_url}/company/{symbol}/consolidated/"
        fundamentals = {}
        try:
            res = requests.get(page_url, headers=self.headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')

            top_ratios = soup.find('ul', id='top-ratios')
            if top_ratios:
                for li in top_ratios.find_all("li"):
                    name_span = li.find("span", class_="name")
                    value_span = li.find("span", class_="number")
                    if name_span and value_span:
                        name = name_span.text.strip()
                        value = value_span.text.strip()
                        if "Market Cap" in name:
                            fundamentals['market_cap'] = safe_float(value) * 10000000
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

            sections = soup.find_all('section', class_='card')
            for section in sections:
                h2 = section.find('h2', class_='card-title')
                if h2 and 'Ratios' in h2.text:
                    for li in section.find_all('li'):
                        name_span = li.find('span', class_='name')
                        value_span = li.find('span', class_='number')
                        if name_span and value_span:
                            name = name_span.text.strip()
                            value = value_span.text.strip()
                            if "Debt to equity" in name: fundamentals['debt_to_equity'] = safe_float(value)
                            if "EPS" in name: fundamentals['eps'] = safe_float(value)
        except Exception as e:
            logger.warning(f"Could not scrape Screener.in for {symbol}: {e}. Will attempt fallback.")

        # --- FALLBACK FOR MISSING DATA ---
        if 'debt_to_equity' not in fundamentals or 'eps' not in fundamentals or fundamentals.get(
                'debt_to_equity') is None or fundamentals.get('eps') is None:
            logger.info(f"Using yfinance fallback for missing fundamentals for {symbol}")
            try:
                ticker_info = yf.Ticker(f"{symbol}.NS").info
                if fundamentals.get('debt_to_equity') is None:
                    fundamentals['debt_to_equity'] = safe_float(ticker_info.get('debtToEquity'))
                if fundamentals.get('eps') is None:
                    fundamentals['eps'] = safe_float(ticker_info.get('trailingEps'))
            except Exception as yf_e:
                logger.error(f"Yfinance fallback failed for {symbol}: {yf_e}")

        return fundamentals

    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetches a comprehensive set of technical indicators using yfinance."""
        yf_symbol = f"{symbol}.NS"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        try:
            stock = yf.Ticker(yf_symbol)
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty: raise ValueError("Yfinance returned empty dataframe")
            close_prices = hist['Close']

            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            exp12 = close_prices.ewm(span=12, adjust=False).mean()
            exp26 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            macd_signal = macd.ewm(span=9, adjust=False).mean()

            return {
                "current_price": safe_float(close_prices.iloc[-1]),
                "rsi": safe_float(rsi.iloc[-1], 50),
                "macd": safe_float(macd.iloc[-1], 0),
                "macd_signal": safe_float(macd_signal.iloc[-1], 0),
                "sma_50": safe_float(close_prices.rolling(window=50).mean().iloc[-1]),
                "sma_200": safe_float(close_prices.rolling(window=200).mean().iloc[-1]),
                "volume": safe_float(hist['Volume'].iloc[-1]),
                "52_week_high": safe_float(close_prices.max()),
                "52_week_low": safe_float(close_prices.min())
            }
        except Exception as e:
            logger.error(f"Yfinance failed for {symbol}: {e}", exc_info=True)
            return self._get_sample_technicals()

    def get_sentiment_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Fetches news and simulates other sentiment data."""
        news_data = self.news_analyzer.analyze_news_sentiment(company_name, symbol)
        news_data['social_media_sentiment'] = round(random.uniform(0.3, 0.9), 2)
        news_data['analyst_rating'] = round(random.uniform(5.5, 9.5), 1)
        return news_data

    def _get_sample_fundamentals(self) -> Dict[str, Any]:
        """Provides sample data if all fetching fails."""
        return {
            'market_cap': 2000000000000, 'current_price': 2500, 'pe_ratio': 25,
            'book_value': 800, 'dividend_yield': 1.5, 'roce': 15, 'roe': 12,
            'debt_to_equity': 0.5, 'eps': 100
        }

    def _get_sample_technicals(self) -> Dict[str, Any]:
        """Provides sample data if all fetching fails."""
        return {
            'current_price': 2500, 'rsi': 55, 'macd': 5, 'macd_signal': 2,
            'sma_50': 2400, 'sma_200': 2200, 'volume': 5000000,
            '52_week_high': 3000, '52_week_low': 2000
        }