import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import os
import random

logger = logging.getLogger(__name__)


class SimpleTransformerModel(nn.Module):
    """Simplified transformer model that matches your architecture"""

    def __init__(self, input_size=35, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        # Simple positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output(x)


class StockPredictionModel:
    def __init__(self):
        # Use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

        self.model_path = os.path.join(data_dir, "model_weight.pt")
        self.features_path = os.path.join(data_dir, "features.pkl")
        self.target_path = os.path.join(data_dir, "target.pkl")
        self.metadata_path = os.path.join(data_dir, "metadata.pkl")

        self.model = None
        self.feature_columns = []
        self.target_columns = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        self.fallback_mode = False
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained model and related data"""
        try:
            # Check if files exist
            for path in [self.model_path, self.features_path, self.target_path, self.metadata_path]:
                if not os.path.exists(path):
                    logger.error(f"File not found: {path}")
                    logger.info("Running in fallback mode with simulated predictions")
                    self.fallback_mode = True
                    return True

            # Load feature information
            self._load_feature_info()

            # Load model weights
            self.model = SimpleTransformerModel(input_size=len(self.feature_columns))

            # Load the state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Load target columns
            with open(self.target_path, 'rb') as f:
                self.target_columns = pickle.load(f)

            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

            self.is_loaded = True
            logger.info(f"Model loaded successfully. Input features: {len(self.feature_columns)}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Running in fallback mode with simulated predictions")
            self.fallback_mode = True
            return True

    def _load_feature_info(self):
        """Load feature information from various sources"""
        try:
            # Try to load from features.pkl
            with open(self.features_path, 'rb') as f:
                features_data = pickle.load(f)

            if hasattr(features_data, 'n_features_in_'):
                # It's a scaler object
                input_size = features_data.n_features_in_
                self.feature_columns = [f'feature_{i}' for i in range(input_size)]
                logger.info(f"Using scaler with {input_size} features")
            elif isinstance(features_data, list):
                # It's a list of feature names
                self.feature_columns = features_data
                logger.info(f"Loaded {len(self.feature_columns)} feature names")
            else:
                # Unknown format, use default features for Indian stocks
                self.feature_columns = [
                    'pe_ratio', 'eps', 'roe', 'debt_to_equity', 'current_ratio',
                    'profit_margin', 'revenue_growth', 'dividend_yield', 'market_cap',
                    'enterprise_value', 'rsi', 'macd', 'sma_50', 'sma_200', 'current_price',
                    'volume', 'price_change', 'news_sentiment', 'social_media_sentiment',
                    'analyst_rating', 'market_sentiment', 'volatility'
                ]
                logger.info(f"Using default {len(self.feature_columns)} feature names")

        except Exception as e:
            logger.error(f"Error loading feature info: {e}")
            # Fallback to default features
            self.feature_columns = [
                'pe_ratio', 'eps', 'roe', 'debt_to_equity', 'current_ratio',
                'profit_margin', 'revenue_growth', 'dividend_yield', 'market_cap',
                'enterprise_value', 'rsi', 'macd', 'sma_50', 'sma_200', 'current_price',
                'volume', 'price_change', 'news_sentiment', 'social_media_sentiment',
                'analyst_rating', 'market_sentiment', 'volatility'
            ]

    def preprocess_input(self, input_data: Dict[str, float]) -> torch.Tensor:
        """Preprocess input data for model prediction"""
        try:
            # Create feature vector in the same order as expected features
            feature_vector = []

            for feature in self.feature_columns:
                if feature in input_data:
                    value = input_data[feature]
                    if pd.isna(value) or np.isnan(value):
                        feature_vector.append(0.0)
                    else:
                        feature_vector.append(float(value))
                else:
                    # Handle missing features with default values
                    if feature in ['pe_ratio', 'rsi', 'sma_50', 'sma_200', 'current_price']:
                        feature_vector.append(0.0)  # Technical defaults
                    elif feature in ['news_sentiment', 'social_media_sentiment', 'market_sentiment']:
                        feature_vector.append(0.5)  # Neutral sentiment
                    elif feature == 'analyst_rating':
                        feature_vector.append(3.0)  # Neutral rating
                    else:
                        feature_vector.append(0.0)

            # Convert to tensor with appropriate shape [batch_size, seq_len, features]
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).unsqueeze(0).to(self.device)
            return input_tensor

        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise

    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using the loaded model or fallback"""
        if self.fallback_mode:
            return self._simulate_prediction(input_data)

        if not self.is_loaded:
            raise Exception("Model not loaded")

        try:
            input_tensor = self.preprocess_input(input_data)

            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction_value = prediction.cpu().numpy()[0][0]

            confidence = max(0.5, min(0.95, 0.8 - abs(prediction_value) * 0.1))

            return {
                "prediction": float(prediction_value),
                "confidence": confidence,
                "features_used": self.feature_columns,
                "model_type": "transformer",
                "features_count": len(self.feature_columns)
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._simulate_prediction(input_data)

    def _simulate_prediction(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate simulated prediction"""
        try:
            pe_ratio = input_data.get('pe_ratio', 15)
            roe = input_data.get('roe', 0.1)
            rsi = input_data.get('rsi', 50)
            news_sentiment = input_data.get('news_sentiment', 0.5)

            prediction = 0.0
            if pe_ratio < 20: prediction += 2.0
            if roe > 0.15: prediction += 3.0
            if 40 < rsi < 60: prediction += 1.5
            if news_sentiment > 0.6: prediction += 2.5
            if pe_ratio > 25: prediction -= 1.5
            if roe < 0.05: prediction -= 2.0
            if rsi > 70: prediction -= 1.0
            if rsi < 30: prediction -= 0.5
            if news_sentiment < 0.4: prediction -= 2.0

            prediction += random.uniform(-2.0, 2.0)
            prediction = max(-10.0, min(10.0, prediction))

            return {
                "prediction": float(prediction),
                "confidence": 0.7,
                "features_used": list(input_data.keys()),
                "model_type": "simulated",
                "features_count": len(input_data)
            }

        except Exception as e:
            return {
                "prediction": random.uniform(-5.0, 5.0),
                "confidence": 0.7,
                "features_used": [],
                "model_type": "fallback",
                "features_count": 0
            }


# Global model instance
prediction_model = StockPredictionModel()


def initialize_model():
    """Initialize the prediction model"""
    global prediction_model
    try:
        success = prediction_model.load_model()
        if success:
            if prediction_model.fallback_mode:
                logger.info("Model initialized in fallback mode")
            else:
                logger.info("Model initialized successfully")
        return success
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False