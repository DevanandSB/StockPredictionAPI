import logging
from datetime import datetime

from googlesearch import search
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# --- HIGHLY FOCUSED KEYWORDS FOR CORPORATE ACTIONS ---
NEWS_FOCUS_KEYWORDS = [
    'dividend announcement', 'bonus shares', 'stock split', 'company acquisition',
    'merger', 'demerger', 'new business launch', 'corporate restructuring'
]

# High-quality news sources
NEWS_SOURCES = [
    "thehindubusinessline.com", "economictimes.indiatimes.com",
    "livemint.com", "business-standard.com", "ndtv.com/business"
]


class NewsSentimentAnalyzer:
    def __init__(self, api_key: str = None):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("VaderSentiment Analyzer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
            self.analyzer = None

    def get_company_news(self, company_name: str, symbol: str) -> list:
        """
        Performs a targeted Google search for corporate action news.
        """
        # Build a query that ONLY looks for the specific keywords you requested
        focus_query = " OR ".join(f'"{k}"' for k in NEWS_FOCUS_KEYWORDS)
        query = f'({company_name} OR {symbol}) ({focus_query}) site:' + " OR site:".join(NEWS_SOURCES)

        articles = []
        try:
            search_results = search(query, num_results=10, sleep_interval=2)
            for url in search_results:
                # --- FIX: Create a natural-looking title, not Title Case ---
                raw_title = url.split('/')[-1].replace('-', ' ').replace('.html', '')
                title = raw_title.capitalize()  # Capitalizes only the first letter

                source = next((s for s in NEWS_SOURCES if s in url), "Unknown Source")

                articles.append({
                    "title": title,
                    "url": url,
                    "source": source,
                    "publishedAt": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "description": "Click to read the full article."
                })
                if len(articles) >= 4:
                    break
        except Exception as e:
            logger.error(f"Error fetching news for {company_name}: {e}")

        return articles

    def analyze_news_sentiment(self, company_name: str, symbol: str) -> dict:
        articles = self.get_company_news(company_name, symbol)
        if not articles or not self.analyzer:
            return {"news_sentiment": 0.5, "articles": []}

        total_sentiment = 0
        for article in articles:
            sentiment_scores = self.analyzer.polarity_scores(article['title'])
            article['sentiment'] = (sentiment_scores['compound'] + 1) / 2
            total_sentiment += article['sentiment']

        average_sentiment = total_sentiment / len(articles) if articles else 0.5
        return {"news_sentiment": average_sentiment, "articles": articles}