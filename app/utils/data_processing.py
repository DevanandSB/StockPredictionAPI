import pandas as pd
import numpy as np
from typing import Dict, List


def validate_input_data(fundamental_data: Dict[str, float],
                        technical_data: Dict[str, float],
                        sentiment_data: Dict[str, float]) -> bool:
    """Validate input data for prediction"""
    # Add validation logic based on your model requirements
    if not all(isinstance(v, (int, float)) for v in list(fundamental_data.values()) +
                                                    list(technical_data.values()) + list(sentiment_data.values())):
        return False

    # Add more validation as needed
    return True


def get_sample_input() -> Dict[str, Dict[str, float]]:
    """Get sample input data for testing"""
    return {
        "fundamental_data": {
            "pe_ratio": 15.2,
            "eps": 2.5,
            "roe": 0.12,
            # Add more fundamental features
        },
        "technical_data": {
            "rsi": 45.6,
            "macd": 0.02,
            "sma_50": 150.25,
            # Add more technical features
        },
        "sentiment_data": {
            "news_sentiment": 0.75,
            "social_media_sentiment": 0.62,
            "analyst_rating": 0.85
            # Add more sentiment features
        }
    }