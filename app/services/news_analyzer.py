import requests
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import nltk
import ssl
import os
import re
import random

logger = logging.getLogger(__name__)

# Fix SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Try to download NLTK data with SSL fix
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Could not download vader_lexicon: {e}")

# Import VADER after attempting download
try:
    from nltk.sentiment import SentimentIntensityAnalyzer

    vader_available = True
except ImportError:
    logger.warning("VADER sentiment analyzer not available")
    vader_available = False


class NewsSentimentAnalyzer:
    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.newsapi_client = None
        self.finbert = None

        # Indian stock symbols mapping
        self.indian_stock_keywords = {
            'RELIANCE': ['Reliance', 'Mukesh Ambani', 'Jio', 'Reliance Industries'],
            'TCS': ['TCS', 'Tata Consultancy', 'Tata Sons'],
            'HDFCBANK': ['HDFC Bank', 'HDFC'],
            'INFY': ['Infosys', 'Narayana Murthy'],
            'ICICIBANK': ['ICICI Bank', 'ICICI'],
        }

        # Initialize VADER if available
        if vader_available:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
            logger.warning("VADER sentiment analyzer not available - using fallback methods")

        # Initialize NewsAPI client if key is provided
        if newsapi_key:
            try:
                self.newsapi_client = NewsApiClient(api_key=newsapi_key)
                logger.info("NewsAPI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NewsAPI client: {e}")

        # Initialize FinBERT model
        try:
            self.finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.finbert = None

    def clean_text(self, text: str) -> str:
        """Clean news text by removing unwanted characters and patterns"""
        if not text: return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_sentiment_finbert(self, text: str) -> float:
        """Analyze sentiment using FinBERT model"""
        if not self.finbert or not text: return 0.5
        try:
            clean_text = self.clean_text(text)[:512]
            result = self.finbert(clean_text)[0]
            if result['label'] == 'positive': return 0.5 + (result['score'] * 0.5)
            if result['label'] == 'negative': return 0.5 - (result['score'] * 0.5)
            return 0.5
        except Exception:
            return 0.5

    def analyze_news_sentiment(self, company_name: str, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles (real or simulated)"""
        articles = self.get_company_news(company_name, symbol)

        if not articles:
            logger.warning(f"No news articles found for {company_name}, using simulated sentiment.")
            return self._get_simulated_sentiment(company_name)

        sentiments = []
        for article in articles:
            # If simulated article has sentiment, use it directly for consistency
            if 'sentiment' in article and article['sentiment'] is not None:
                sentiments.append(article['sentiment'])
            else:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title}. {description}"
                if self.finbert:
                    sentiment_score = self.analyze_sentiment_finbert(text)
                elif self.vader_analyzer:
                    sentiment_score = (self.vader_analyzer.polarity_scores(text)['compound'] + 1) / 2
                else:
                    sentiment_score = 0.5
                article['sentiment'] = sentiment_score
                sentiments.append(sentiment_score)

        if not sentiments: return self._get_simulated_sentiment(company_name)

        avg_sentiment = np.mean(sentiments)
        return {
            'news_sentiment': float(avg_sentiment),
            'social_media_sentiment': float(max(0.3, min(0.9, avg_sentiment + random.uniform(-0.1, 0.1)))),
            'analyst_rating': float(max(2.5, min(4.8, 3.5 + (avg_sentiment - 0.5) * 3))),
            'market_sentiment': float(avg_sentiment),
            'volatility': float(max(0.1, min(0.5, 0.3 + abs(avg_sentiment - 0.5) * 0.4))),
            'articles': articles[:5]
        }

    def get_company_news(self, company_name: str, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Fetch news articles for a company, falling back to simulation."""
        if not self.newsapi_client:
            logger.warning("NewsAPI client not available - using simulated news.")
            return self._get_simulated_news(company_name, symbol)

        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            keywords = self.indian_stock_keywords.get(symbol, [company_name])
            query = " OR ".join(f'"{k}"' for k in keywords) + f" AND (stock OR market OR India)"

            response = self.newsapi_client.get_everything(
                q=query, from_param=from_date.strftime('%Y-%m-%d'), language='en',
                sort_by='relevancy', page_size=20)

            articles = response.get('articles', [])
            if not articles:
                return self._get_simulated_news(company_name, symbol)

            # Format articles
            formatted_articles = []
            for art in articles:
                formatted_articles.append({
                    'title': art.get('title'),
                    'url': art.get('url'),
                    'published_at': art.get('publishedAt'),
                    'source': art.get('source', {}).get('name'),
                    'preview': art.get('description') or art.get('content', '') or '',
                })
            return formatted_articles
        except Exception as e:
            logger.error(f"Error fetching news for {company_name}: {e}")
            return self._get_simulated_news(company_name, symbol)

    def _get_simulated_news(self, company_name: str, symbol: str) -> List[Dict[str, Any]]:
        """Generate more realistic and varied simulated news articles."""
        templates = [
            ("positive", f"{company_name} reports record quarterly profits, beating analyst expectations.", 0.85),
            ("positive", f"New strategic partnership set to boost {company_name}'s market share significantly.", 0.8),
            ("positive", f"Analysts upgrade {symbol} to 'Strong Buy' following positive growth outlook.", 0.75),
            ("neutral", f"{company_name} to announce future strategy at upcoming investor meeting.", 0.5),
            ("neutral", f"Market shows mixed reaction to {company_name}'s latest product announcement.", 0.45),
            ("negative", f"Regulatory concerns cast a shadow over {company_name}'s expansion plans.", 0.25),
            ("negative", f"{company_name} misses revenue targets amid rising operational costs.", 0.2),
        ]

        articles = []
        for _ in range(random.randint(4, 6)):
            category, title, sentiment = random.choice(templates)
            sentiment += random.uniform(-0.05, 0.05)  # Add jitter
            articles.append({
                'title': title,
                'preview': f"This is a simulated news article regarding {company_name}. Developments are being closely watched by investors.",
                'url': '#',
                'published_at': (datetime.now() - timedelta(days=random.randint(1, 15))).isoformat() + 'Z',
                'source': random.choice(['Simulated Business Times', 'Fake Market Watch']),
                'sentiment': sentiment
            })
        return articles

    def _get_simulated_sentiment(self, company_name: str) -> Dict[str, Any]:
        """Generate simulated sentiment data based on simulated news."""
        articles = self._get_simulated_news(company_name, "")
        sentiments = [art['sentiment'] for art in articles]
        avg_sentiment = np.mean(sentiments)

        return {
            'news_sentiment': float(avg_sentiment),
            'social_media_sentiment': float(max(0.3, min(0.9, avg_sentiment + random.uniform(-0.1, 0.1)))),
            'analyst_rating': float(max(2.5, min(4.8, 3.5 + (avg_sentiment - 0.5) * 3))),
            'market_sentiment': float(avg_sentiment),
            'volatility': float(max(0.1, min(0.5, 0.3 + abs(avg_sentiment - 0.5) * 0.4))),
            'articles': articles[:5]
        }


# Global analyzer instance
try:
    newsapi_key = os.getenv('NEWSAPI_KEY')
    news_analyzer = NewsSentimentAnalyzer(newsapi_key=newsapi_key)
except Exception as e:
    logger.error(f"Failed to initialize NewsSentimentAnalyzer: {e}")
    news_analyzer = NewsSentimentAnalyzer()