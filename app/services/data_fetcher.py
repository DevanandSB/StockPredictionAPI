import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """Safely convert value to float, handling NaN and None"""
    if value is None or pd.isna(value) or np.isnan(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class DataFetcher:
    def __init__(self, newsapi_key: Optional[str] = None):
        self.screener_base_url = "https://www.screener.in/api/company"
        self.newsapi_key = newsapi_key
        self.news_analyzer = None

        # Try to import and initialize news analyzer, but fallback if it fails
        try:
            from app.services.news_analyzer import NewsSentimentAnalyzer
            self.news_analyzer = NewsSentimentAnalyzer(newsapi_key)
            logger.info("News analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize news analyzer: {e}")
            # Create a fallback analyzer
            self.news_analyzer = self.FallbackAnalyzer()

    class FallbackAnalyzer:
        """Fallback analyzer when main analyzer fails"""

        def analyze_news_sentiment(self, company_name, symbol):
            return {
                'news_sentiment': 0.5,
                'social_media_sentiment': 0.5,
                'analyst_rating': 3.5,
                'market_sentiment': 0.5,
                'volatility': 0.3,
                'articles': []
            }

        def get_company_news(self, company_name, symbol):
            return []

    def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from Yahoo Finance"""
        try:
            # Try Yahoo Finance
            stock = yf.Ticker(f"{symbol}.NS")  # .NS for NSE
            info = stock.info

            fundamentals = {
                "pe_ratio": safe_float(info.get('trailingPE', 15)),
                "eps": safe_float(info.get('trailingEps', 2.5)),
                "roe": safe_float(info.get('returnOnEquity', 0.12)),
                "debt_to_equity": safe_float(info.get('debtToEquity', 0.6)),
                "current_ratio": safe_float(info.get('currentRatio', 1.8)),
                "profit_margin": safe_float(info.get('profitMargins', 0.15)),
                "revenue_growth": safe_float(info.get('revenueGrowth', 0.08)),
                "dividend_yield": safe_float(info.get('dividendYield', 0.025)),
                "market_cap": safe_float(info.get('marketCap', 5000000000)),
                "enterprise_value": safe_float(info.get('enterpriseValue', 5500000000))
            }

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return self._get_sample_fundamentals()

    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetch technical indicators using Yahoo Finance"""
        try:
            stock = yf.Ticker(f"{symbol}.NS")

            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty or len(hist) < 50:
                return self._get_sample_technicals()

            # Calculate technical indicators
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26

            # Calculate moving averages
            sma_50 = hist['Close'].rolling(window=50).mean()
            sma_200 = hist['Close'].rolling(window=200).mean()

            # Get latest values, handling NaN
            latest_rsi = safe_float(rsi.iloc[-1] if not rsi.empty else 50)
            latest_macd = safe_float(macd.iloc[-1] if not macd.empty else 0)
            latest_sma_50 = safe_float(sma_50.iloc[-1] if not sma_50.empty else hist['Close'].iloc[-1])
            latest_sma_200 = safe_float(sma_200.iloc[-1] if not sma_200.empty else hist['Close'].iloc[-1])
            current_price = safe_float(hist['Close'].iloc[-1] if not hist.empty else 0)
            volume = safe_float(hist['Volume'].iloc[-1] if not hist.empty else 0)

            # Calculate price change safely
            if len(hist) > 1:
                price_change = safe_float(
                    (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100)
            else:
                price_change = 0.0

            technicals = {
                "rsi": latest_rsi,
                "macd": latest_macd,
                "sma_50": latest_sma_50,
                "sma_200": latest_sma_200,
                "current_price": current_price,
                "volume": volume,
                "price_change": price_change
            }

            return technicals

        except Exception as e:
            logger.error(f"Error fetching technicals for {symbol}: {e}")
            return self._get_sample_technicals()

    def get_sentiment_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Get sentiment data using NewsAPI and FinBERT"""
        try:
            sentiment_data = self.news_analyzer.analyze_news_sentiment(company_name, symbol)
            # Ensure all values are native Python types (not numpy)
            return {
                'news_sentiment': float(sentiment_data.get('news_sentiment', 0.5)),
                'social_media_sentiment': float(sentiment_data.get('social_media_sentiment', 0.5)),
                'analyst_rating': float(sentiment_data.get('analyst_rating', 3.5)),
                'market_sentiment': float(sentiment_data.get('market_sentiment', 0.5)),
                'volatility': float(sentiment_data.get('volatility', 0.3)),
                'articles': sentiment_data.get('articles', [])
            }
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return self._get_sample_sentiment()

    def _get_sample_fundamentals(self) -> Dict[str, Any]:
        """Return sample fundamental data"""
        return {
            "pe_ratio": 15.2,
            "eps": 2.5,
            "roe": 0.12,
            "debt_to_equity": 0.6,
            "current_ratio": 1.8,
            "profit_margin": 0.15,
            "revenue_growth": 0.08,
            "dividend_yield": 0.025,
            "market_cap": 5000000000,
            "enterprise_value": 5500000000
        }

    def _get_sample_technicals(self) -> Dict[str, Any]:
        """Return sample technical data"""
        return {
            "rsi": 45.6,
            "macd": 0.02,
            "sma_50": 150.25,
            "sma_200": 145.80,
            "current_price": 152.30,
            "volume": 2500000,
            "price_change": 1.5
        }

    def _get_sample_sentiment(self) -> Dict[str, Any]:
        """Return sample sentiment data"""
        return {
            "news_sentiment": 0.75,
            "social_media_sentiment": 0.62,
            "analyst_rating": 4.2,
            "market_sentiment": 0.68,
            "volatility": 0.25,
            "articles": []
        }


# Global data fetcher instance - initialize with NewsAPI key from environment
try:
    newsapi_key = os.getenv('NEWSAPI_KEY')
    data_fetcher = DataFetcher(newsapi_key=newsapi_key)
except Exception as e:
    logger.error(f"Failed to initialize DataFetcher: {e}")
    # Create fallback instance
    data_fetcher = DataFetcher()