# app/services/__init__.py

import os
from .data_fetcher import DataFetcher

# This line creates the single instance of DataFetcher that the app will import and use.
data_fetcher = DataFetcher(newsapi_key=os.getenv('NEWSAPI_KEY'))