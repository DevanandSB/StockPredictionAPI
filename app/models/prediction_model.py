import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import os
import random
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class StockPredictionModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        self.metadata_path = os.path.join(data_dir, "metadata.pkl")
        self.metadata = None
        self.feature_columns = []
        self.is_loaded = False

    def load_model(self):
        """Loads model metadata to prepare for simulation."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                    self.feature_columns = self.metadata.get('feature_columns', [])
            self.is_loaded = True
            logger.info("Model ready for simulated predictions.")
            return True
        except Exception as e:
            logger.error(f"Error loading model info: {e}")
            return False

    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Provides a single, short-term prediction."""
        horizons = self.predict_horizons(input_data)
        short_term_prediction = horizons.get('next_day', {})
        return {
            "prediction": short_term_prediction.get('prediction', 0),
            "confidence": short_term_prediction.get('confidence', 0.5),
            "model_type": "simulated_heuristic_model",
        }

    def predict_horizons(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates simulated predictions for different time horizons
        by weighing different types of data.
        """
        # --- Factors Extraction ---
        # Technicals (for short-term)
        rsi = data.get('rsi', 50)
        price_change = data.get('price_change', 0)

        # Sentiment (for short-to-medium term)
        news_sentiment = data.get('news_sentiment', 0.5)

        # Fundamentals (for long-term)
        pe = data.get('pe_ratio', 25)
        roe = data.get('roe', 0.10)
        debt_to_equity = data.get('debt_to_equity', 1.0)
        revenue_growth = data.get('revenue_growth', 0.05)

        # --- Prediction Logic ---
        # Next Day: Heavily reliant on technicals and sentiment
        day_pred = (50 - rsi) * 0.1 + price_change * 0.5 + (news_sentiment - 0.5) * 4

        # Next Month: Sentiment and fundamental momentum
        month_pred = (news_sentiment - 0.5) * 5 + revenue_growth * 50 + (0.15 - roe) * -20

        # Next Year: Primarily strong fundamentals
        year_pred = (0.20 - roe) * -100 + (20 - pe) * 0.5 + revenue_growth * 100 + (1 - debt_to_equity) * 5

        # Next 2 Years: Strong fundamentals and stability
        two_year_pred = year_pred * 2.2 + (0.18 - roe) * -150  # Compounded growth effect

        # --- Packaging Results ---
        return {
            "next_day": self._format_prediction(day_pred, 0.85, "Technical & Sentiment"),
            "next_month": self._format_prediction(month_pred, 0.75, "Sentiment & Momentum"),
            "next_year": self._format_prediction(year_pred, 0.65, "Fundamental Strength"),
            "next_2_years": self._format_prediction(two_year_pred, 0.55, "Long-Term Fundamentals"),
        }

    def _format_prediction(self, raw_pred: float, confidence: float, basis: str) -> Dict[str, Any]:
        """Helper to format and cap the prediction results."""
        capped_pred = max(-99, min(200, raw_pred + random.uniform(-1, 1)))
        return {
            "prediction_percent": round(capped_pred, 2),
            "confidence": round(confidence + random.uniform(-0.05, 0.05), 2),
            "basis": basis
        }


# Global model instance
prediction_model = StockPredictionModel()


def initialize_model():
    """Initialize the prediction model"""
    global prediction_model
    return prediction_model.load_model()