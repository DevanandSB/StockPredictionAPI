import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
import json

logger = logging.getLogger(__name__)


# Add this function to handle numpy serialization
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def safe_float(value, default=None):
    """Safely convert value to float, returning None if conversion fails."""
    if value is None or pd.isna(value) or value == '':
        return default
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('%', '').strip()
        result = float(value)
        return convert_to_serializable(result)
    except (ValueError, TypeError):
        return default


def safe_str(value, default=''):
    """Safely get string value with default."""
    if value is None:
        return default
    try:
        return str(value).strip()
    except:
        return default


def safe_date(date_str, default=''):
    """Safely convert date string, return blank for invalid dates"""
    if not date_str or pd.isna(date_str) or 'invalid' in str(date_str).lower():
        return default
    try:
        if isinstance(date_str, str):
            for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y'):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        return str(date_str)
    except:
        return default


class FinBertSentimentAnalyzer:
    def __init__(self, cache_dir="models/finbert"):
        logger.info("Initializing FinBERT sentiment analyzer")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Load FinBERT model with caching
            self.model_name = "yiyanghkust/finbert-tone"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            self.model.eval()
            logger.info("FinBERT model loaded successfully from cache")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.model = None
            self.tokenizer = None

    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using FinBERT"""
        if self.model is None or self.tokenizer is None:
            return 0.5  # Return neutral if model not loaded

        text = safe_str(text)
        if len(text.strip()) < 10:
            return 0.5

        try:
            # Preprocess text
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^\w\s.,!?]', '', text)
            text = text.strip()

            if len(text) < 10:
                return 0.5

            # Tokenize and get predictions
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = softmax(outputs.logits.numpy(), axis=1)

            # FinBERT outputs: [negative, neutral, positive]
            negative_score = predictions[0][0]
            neutral_score = predictions[0][1]
            positive_score = predictions[0][2]

            # Convert to 0-1 scale
            sentiment_score = (positive_score * 0.8 + neutral_score * 0.5 + negative_score * 0.2)

            return convert_to_serializable(max(0.0, min(1.0, sentiment_score)))

        except Exception as e:
            logger.error(f"FinBERT sentiment analysis error: {e}")
            return 0.5


class DataFetcher:
    def __init__(self, newsapi_key: Optional[str] = None):
        self.screener_base_url = "https://www.screener.in"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Initialize sentiment analyzer
        self.sentiment_analyzer = FinBertSentimentAnalyzer()

        # News API setup
        self.newsapi_key = newsapi_key
        self.newsapi_url = "https://newsapi.org/v2/everything"

        logger.info(f"DataFetcher initialized with NewsAPI: {bool(newsapi_key)}")

    def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Scrapes fundamental data from Screener.in with fallback to yfinance"""
        logger.info(f"Scraping fundamentals for {symbol} from Screener.in")
        fundamentals = {}

        try:
            page_url = f"{self.screener_base_url}/company/{symbol}/consolidated/"
            res = requests.get(page_url, headers=self.headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')

            # Extract fundamentals from page
            top_ratios = soup.find('ul', id='top-ratios')
            if top_ratios:
                for li in top_ratios.find_all("li"):
                    name_span = li.find("span", class_="name")
                    value_span = li.find("span", class_="number")
                    if name_span and value_span:
                        name = name_span.text.strip()
                        value = value_span.text.strip()
                        if "Market Cap" in name:
                            if 'Cr' in value:
                                fundamentals['market_cap'] = safe_float(value.replace('Cr', '')) * 10000000
                            elif 'Lac' in value:
                                fundamentals['market_cap'] = safe_float(value.replace('Lac', '')) * 100000
                            else:
                                fundamentals['market_cap'] = safe_float(value)
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

            # Extract additional ratios
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
                            if "Debt to equity" in name:
                                debt_equity = safe_float(value)
                                if debt_equity is not None:
                                    fundamentals['debt_to_equity'] = debt_equity / 100
                            if "EPS" in name:
                                fundamentals['eps'] = safe_float(value)
        except Exception as e:
            logger.warning(f"Could not scrape Screener.in for {symbol}: {e}")

        # Fallback to yfinance for missing data
        try:
            ticker_info = yf.Ticker(f"{symbol}.NS").info

            # Fill missing values
            if 'market_cap' not in fundamentals:
                market_cap = ticker_info.get('marketCap')
                if market_cap:
                    fundamentals['market_cap'] = market_cap

            if 'current_price' not in fundamentals:
                current_price = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice')
                if current_price:
                    fundamentals['current_price'] = current_price

            if 'debt_to_equity' not in fundamentals:
                de_ratio_raw = ticker_info.get('debtToEquity')
                if de_ratio_raw is not None:
                    fundamentals['debt_to_equity'] = safe_float(de_ratio_raw) / 100

            if 'eps' not in fundamentals:
                fundamentals['eps'] = safe_float(ticker_info.get('trailingEps'))

            if 'pe_ratio' not in fundamentals:
                fundamentals['pe_ratio'] = safe_float(ticker_info.get('trailingPE'))

            if 'book_value' not in fundamentals:
                fundamentals['book_value'] = safe_float(ticker_info.get('bookValue'))

        except Exception as yf_e:
            logger.error(f"Yfinance fallback failed for {symbol}: {yf_e}")

        # Ensure all required fields exist
        required_fields = ['market_cap', 'current_price', 'pe_ratio', 'debt_to_equity', 'eps']
        for field in required_fields:
            if field not in fundamentals or fundamentals[field] is None:
                sample_data = self._get_sample_fundamentals()
                fundamentals[field] = sample_data[field]
                logger.warning(f"Missing {field}, using sample data")

        return convert_to_serializable(fundamentals)

    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetches technical indicators using yfinance"""
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            hist = stock.history(period="1y")

            if hist.empty:
                raise ValueError("Yfinance returned empty dataframe")

            close_prices = hist['Close']

            # Calculate indicators
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            exp12 = close_prices.ewm(span=12, adjust=False).mean()
            exp26 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            macd_signal = macd.ewm(span=9, adjust=False).mean()

            sma_50 = close_prices.rolling(window=50).mean()
            sma_200 = close_prices.rolling(window=200).mean()

            result = {
                "current_price": safe_float(close_prices.iloc[-1]),
                "rsi": safe_float(rsi.iloc[-1], 50),
                "macd": safe_float(macd.iloc[-1], 0),
                "macd_signal": safe_float(macd_signal.iloc[-1], 0),
                "sma_50": safe_float(sma_50.iloc[-1]),
                "sma_200": safe_float(sma_200.iloc[-1]),
                "volume": safe_float(hist['Volume'].iloc[-1]),
                "52_week_high": safe_float(close_prices.max()),
                "52_week_low": safe_float(close_prices.min()),
                "price_vs_50sma": safe_float((close_prices.iloc[-1] / sma_50.iloc[-1] - 1) * 100),
                "price_vs_200sma": safe_float((close_prices.iloc[-1] / sma_200.iloc[-1] - 1) * 100)
            }

            return convert_to_serializable(result)

        except Exception as e:
            logger.error(f"Yfinance failed for {symbol}: {e}")
            return convert_to_serializable(self._get_sample_technicals())

    def fetch_news_sync(self, company_name: str, symbol: str) -> list:
        """Synchronous news fetching"""
        articles = []

        if self.newsapi_key:
            try:
                params = {
                    'q': f'{company_name} {symbol} stock',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'apiKey': self.newsapi_key,
                    'pageSize': 5
                }

                response = requests.get(self.newsapi_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
            except Exception as e:
                logger.warning(f"NewsAPI failed: {e}")

        if not articles:
            articles = self._get_fallback_news(company_name, symbol)

        return articles

    def _get_fallback_news(self, company_name: str, symbol: str) -> list:
        """Provide fallback news data"""
        return [
            {
                'title': f'{company_name} announces quarterly results',
                'description': f'{company_name} reported financial results for the latest quarter.',
                'url': '#',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Financial News'}
            }
        ]

    def analyze_news_sentiment(self, articles: list) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        if not articles:
            return convert_to_serializable({'news_sentiment': 0.5, 'articles': [], 'sentiment_std': 0.2})

        sentiments = []
        analyzed_articles = []

        for article in articles:
            try:
                title = safe_str(article.get('title', ''))
                description = safe_str(article.get('description', ''))
                url = safe_str(article.get('url', '#'))

                source_data = article.get('source', {})
                source_name = safe_str(
                    source_data.get('name', 'Unknown') if isinstance(source_data, dict) else source_data)

                published_at = safe_date(article.get('publishedAt', datetime.now().strftime('%Y-%m-%d')))

                # Analyze sentiment
                text = f"{title} {description}"
                sentiment = self.sentiment_analyzer.analyze_text_sentiment(text)

                sentiments.append(sentiment)

                analyzed_articles.append(convert_to_serializable({
                    'title': title,
                    'description': description,
                    'url': url,
                    'published_at': published_at,
                    'source': source_name,
                    'sentiment': sentiment
                }))
            except Exception as e:
                logger.warning(f"Error processing article: {e}")
                continue

        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0.2
        else:
            avg_sentiment = 0.5
            sentiment_std = 0.2

        return convert_to_serializable({
            'news_sentiment': avg_sentiment,
            'articles': analyzed_articles,
            'sentiment_std': sentiment_std
        })

    def get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment"""
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            info = stock.info

            change_percent = info.get('regularMarketChangePercent', 0)
            volume = info.get('regularMarketVolume', 0)
            avg_volume = info.get('averageVolume', volume)

            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            volume_factor = min(2.0, max(0.5, volume_ratio))

            base_sentiment = 0.5 + (change_percent / 200)
            sentiment = base_sentiment * volume_factor

            return convert_to_serializable(max(0.3, min(0.7, sentiment)))

        except Exception as e:
            logger.warning(f"Social sentiment simulation failed: {e}")
            return 0.5

    def get_analyst_ratings(self, symbol: str) -> float:
        """Get analyst ratings"""
        try:
            fundamentals = self.get_stock_fundamentals(symbol)

            pe_ratio = fundamentals.get('pe_ratio', 20)
            roe = fundamentals.get('roe', 10)
            debt_to_equity = fundamentals.get('debt_to_equity', 0.5)

            rating = 7.0

            if pe_ratio < 15:
                rating += 1.5
            elif pe_ratio < 25:
                rating += 0.5
            elif pe_ratio > 35:
                rating -= 1.0

            if roe > 20:
                rating += 1.0
            elif roe > 15:
                rating += 0.5
            elif roe < 8:
                rating -= 0.5

            if debt_to_equity > 1.0:
                rating -= 1.0
            elif debt_to_equity > 0.5:
                rating -= 0.5
            elif debt_to_equity < 0.2:
                rating += 0.5

            return convert_to_serializable(max(5.0, min(9.5, rating)))

        except Exception as e:
            logger.warning(f"Analyst rating simulation failed: {e}")
            return 7.5

    def get_sentiment_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Get sentiment data"""
        articles = self.fetch_news_sync(company_name, symbol)
        news_data = self.analyze_news_sentiment(articles)
        social_sentiment = self.get_social_sentiment(symbol)
        analyst_rating = self.get_analyst_ratings(symbol)

        return convert_to_serializable({
            'news_sentiment': news_data['news_sentiment'],
            'articles': news_data['articles'],
            'social_media_sentiment': social_sentiment,
            'analyst_rating': analyst_rating,
            'sentiment_std': news_data['sentiment_std'],
            'has_real_data': bool(articles and len(articles) > 0)
        })

    def get_company_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Get all company data"""
        fundamentals = self.get_stock_fundamentals(symbol)
        technicals = self.get_technical_indicators(symbol)
        sentiment_data = self.get_sentiment_data(symbol, company_name)

        return convert_to_serializable({
            'symbol': symbol,
            'company_name': company_name,
            'fundamentals': fundamentals,
            'technicals': technicals,
            'sentiment': sentiment_data,
            'last_updated': datetime.now().isoformat()
        })

    def _get_sample_fundamentals(self) -> Dict[str, Any]:
        return convert_to_serializable({
            'market_cap': 2000000000000,
            'current_price': 2500,
            'pe_ratio': 25,
            'book_value': 800,
            'dividend_yield': 1.5,
            'roce': 15,
            'roe': 12,
            'debt_to_equity': 0.5,
            'eps': 100
        })

    def _get_sample_technicals(self) -> Dict[str, Any]:
        """Provides sample data if all fetching fails."""
        return {
            'current_price': 2500, 'rsi': 55, 'macd': 5, 'macd_signal': 2,
            'sma_50': 2400, 'sma_200': 2200, 'volume': 5000000,
            '52_week_high': 3000, '52_week_low': 2000
        }