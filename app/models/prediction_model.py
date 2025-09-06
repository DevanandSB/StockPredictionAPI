import torch
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, AsyncGenerator, List, Union
import logging
import os
import asyncio
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        torch.nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StockPredictionTransformer(torch.nn.Module):
    def __init__(self, feature_size, d_model=128, nhead=4, num_encoder_layers=3,
                 dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1000)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output(x)
        return x


class RealPredictionModel:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.scaler = None
        self.is_loaded = False
        self.feature_columns = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(base_dir)
        self.model_path = os.path.join(app_dir, "data", "model_weight.pt")
        self.metadata_path = os.path.join(app_dir, "data", "metadata.pkl")
        self.scaler_path = os.path.join(app_dir, "data", "scaler.pkl")

        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Metadata path: {self.metadata_path}")
        logger.info(f"Scaler path: {self.scaler_path}")

    def load_model(self) -> bool:
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                logger.error(f"Model weights not found at: {self.model_path}")
                return False
            if not os.path.exists(self.metadata_path):
                logger.error(f"Metadata not found at: {self.metadata_path}")
                return False
            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler not found at: {self.scaler_path}")
                return False

            logger.info("✅ All model files found!")

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            # Load scaler - handle both scaler object and dictionary format
            with open(self.scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)

            # Handle different scaler formats
            if isinstance(scaler_data, dict):
                if 'feature_scaler' in scaler_data:
                    self.scaler = scaler_data['feature_scaler']
                    logger.info("Loaded scaler from dictionary: feature_scaler")
                elif 'scaler' in scaler_data:
                    self.scaler = scaler_data['scaler']
                    logger.info("Loaded scaler from dictionary: scaler")
                else:
                    scaler_found = False
                    for key, value in scaler_data.items():
                        if hasattr(value, 'transform') and hasattr(value, 'fit'):
                            self.scaler = value
                            logger.info(f"Loaded scaler from dictionary key: {key}")
                            scaler_found = True
                            break

                    if not scaler_found:
                        logger.error("Could not find scaler object in dictionary")
                        return False
            else:
                self.scaler = scaler_data
                logger.info("Loaded scaler object directly")

            self.feature_columns = self.metadata.get('feature_columns', [])
            logger.info(f"Feature columns: {self.feature_columns}")
            logger.info(f"Number of features: {len(self.feature_columns)}")

            # Initialize model
            model_config = self.metadata.get('model_config', {})

            d_model = model_config.get('hidden_dim', 128)
            nhead = model_config.get('num_heads', 4)
            num_encoder_layers = model_config.get('num_layers', 3)
            dropout = model_config.get('dropout', 0.1)

            logger.info(
                f"Model config: d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, dropout={dropout}")

            self.model = StockPredictionTransformer(
                feature_size=len(self.feature_columns),
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout
            )

            # Load trained weights
            logger.info("Loading model weights...")
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True

            logger.info("✅ Real Transformer prediction model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading real model: {e}", exc_info=True)
            return False

    async def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch real-time market data"""
        try:
            if not any(ext in symbol for ext in ['.NS', '.BO']) and not '.' in symbol:
                symbol += '.NS'

            stock = yf.Ticker(symbol)
            hist = stock.history(period="2y", interval="1d")  # Fetch 2 years for better volatility calc

            if hist.empty:
                base_symbol = symbol.split('.')[0]
                stock = yf.Ticker(base_symbol)
                hist = stock.history(period="2y", interval="1d")

            if hist.empty:
                logger.error(f"Could not fetch data for {symbol}")
                return pd.DataFrame()

            return self._calculate_technical_indicators(hist)

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in market data")
                    return pd.DataFrame()

            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['price_change'] = df['Close'].diff()
            df['volatility_20'] = df['Returns'].rolling(window=20).std()
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20'].replace(0, 1)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['date'] = df.index.strftime('%Y-%m-%d')

            return df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def _preprocess_features(self, input_features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features before scaling"""
        processed_features = input_features.copy()

        # Convert date columns to numerical values
        date_columns = ['date', 'Date_fund']
        for col in date_columns:
            if col in processed_features.columns:
                try:
                    if col == 'date':
                        dates = pd.to_datetime(processed_features[col])
                    else:
                        dates = pd.to_datetime(processed_features[col], errors='coerce')
                        dates = dates.fillna(pd.to_datetime('today'))

                    reference_date = pd.to_datetime('2000-01-01')
                    processed_features[col] = (dates - reference_date).dt.days

                except Exception as e:
                    logger.warning(f"Error processing date column {col}: {e}")
                    processed_features[col] = 10000

        # Convert sector to numerical
        if 'Sector' in processed_features.columns:
            sector_mapping = {'Diversified': 1, 'IT': 2, 'Finance': 3, 'Healthcare': 4}
            processed_features['Sector'] = processed_features['Sector'].map(
                lambda x: sector_mapping.get(x, 1)
            )

        # Ensure all values are numeric
        for col in processed_features.columns:
            if processed_features[col].dtype == 'object':
                try:
                    processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')
                except:
                    processed_features[col] = 0

        return processed_features.fillna(0)

    def _prepare_fundamental_features(self, symbol: str, current_price: float,
                                      company_data: Dict = None) -> pd.DataFrame:
        """Prepare fundamental features"""
        try:
            if company_data:
                fundamentals = {
                    'Date_fund': datetime.now().strftime('%Y-%m-%d'),
                    'Market Cap': company_data.get('market_cap', 1000000),
                    'Current Price': current_price,
                    'High_fund': company_data.get('52_week_high', current_price * 1.1),
                    'Low_fund': company_data.get('52_week_low', current_price * 0.9),
                    'Stock P/E': company_data.get('pe_ratio', 25.0),
                    'Book Value': company_data.get('book_value', current_price / 3.0),
                    'Dividend Yield': company_data.get('dividend_yield', 2.5),
                    'ROE': company_data.get('roe', 15.0),
                    'EPS': company_data.get('eps', current_price / 25.0),
                    'Debt to equity': company_data.get('debt_to_equity', 0.5),
                    'Price to book value': round(current_price / company_data.get('book_value', current_price / 3.0),
                                                 2),
                    'Sector': 'Diversified',
                    'Volume_fund': company_data.get('volume', 1000000),
                    'avg_sentiment': company_data.get('news_sentiment', 0.5),
                    'news_count': len(company_data.get('articles', [])),
                    'sentiment_std': 0.2,
                    'has_news': 1 if company_data.get('articles') else 0,
                    'Dividends': 0,
                    'Stock Splits': 0,
                    'target_return': 0.0
                }
            else:  # Fallback with dummy data if none is provided
                fundamentals = {
                    'Date_fund': datetime.now().strftime('%Y-%m-%d'), 'Market Cap': 1000000,
                    'Current Price': current_price, 'High_fund': current_price * 1.1,
                    'Low_fund': current_price * 0.9, 'Stock P/E': 25.0,
                    'Book Value': current_price / 3.0, 'Dividend Yield': 2.5,
                    'ROE': 15.0, 'EPS': current_price / 25.0,
                    'Debt to equity': 0.5, 'Price to book value': 3.0,
                    'Sector': 'Diversified', 'Volume_fund': 1000000,
                    'avg_sentiment': 0.5, 'news_count': 5, 'sentiment_std': 0.2,
                    'has_news': 1, 'Dividends': 0, 'Stock Splits': 0, 'target_return': 0.0
                }
            return pd.DataFrame([fundamentals])
        except Exception as e:
            logger.error(f"Error preparing fundamental features: {e}")
            return pd.DataFrame()

    def _prepare_input_features(self, market_data: pd.DataFrame, current_price: float,
                                symbol: str, company_data: Dict = None) -> torch.Tensor:
        """Prepare input features for the model"""
        if market_data.empty:
            raise ValueError("No market data available")

        latest_market = market_data.iloc[-1:].copy()
        fundamental_data = self._prepare_fundamental_features(symbol, current_price, company_data)

        combined_data = latest_market.copy()
        for col in fundamental_data.columns:
            if col in self.feature_columns:
                combined_data[col] = fundamental_data[col].iloc[0]

        combined_data['Current Price'] = current_price
        combined_data['Close'] = current_price

        for col in self.feature_columns:
            if col not in combined_data.columns:
                combined_data[col] = 0.0

        input_features = combined_data[self.feature_columns].fillna(0.0)
        processed_features = self._preprocess_features(input_features)

        scaled_features = self.scaler.transform(processed_features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1)

        return input_tensor

    async def predict_stream(self, symbol: Union[str, Dict], current_price: float = None) -> AsyncGenerator[Dict, Any]:
        if not self.is_loaded:
            yield {"error": "Model not loaded.", "progress": 0};
            return

        try:
            company_data = symbol if isinstance(symbol, dict) else {}
            symbol_str = company_data.get('symbol', symbol)

            yield {"status": "Fetching market data...", "progress": 10}
            market_data = await self._fetch_market_data(symbol_str)
            if market_data.empty:
                yield {"error": f"Could not fetch data for {symbol_str}"};
                return

            current_price = market_data['Close'].iloc[-1]

            yield {"status": "Preparing features...", "progress": 25}
            input_tensor = self._prepare_input_features(market_data, current_price, symbol_str, company_data)

            yield {"status": "Running AI prediction...", "progress": 40}
            await asyncio.sleep(0.1)

            with torch.no_grad():
                model_prediction = self.model(input_tensor).item()

            async for result in self.run_monte_carlo_simulation(current_price, model_prediction, market_data):
                yield result

        except Exception as e:
            logger.error(f"Error in prediction stream: {e}", exc_info=True)
            yield {"error": f"Prediction failed: {str(e)}"}

    async def run_monte_carlo_simulation(self, current_price: float, model_prediction: float,
                                         market_data: pd.DataFrame) -> AsyncGenerator[Dict, Any]:
        """Runs a detailed, day-by-day Monte Carlo simulation using Geometric Brownian Motion."""
        returns = market_data['Returns'].dropna()
        daily_volatility = returns.std() if len(returns) > 1 else 0.02
        annual_volatility = daily_volatility * np.sqrt(252)

        time_horizons = {
            "Next Day": 1, "Next Week": 5, "Next 15 Days": 15,
            "Next 30 Days": 30, "Next 3 Months": 63, "Next 6 Months": 126,
            "Next Year": 252
        }

        num_simulations = 5000  # Increased for higher fidelity
        max_days = max(time_horizons.values())

        # The AI's influence provides a "drift" for the random walk. It fades over the full year.
        model_influence = max(0.05, 1 - (max_days / 365))
        drift = (model_prediction * model_influence)

        # Calculate daily drift and volatility for the GBM formula
        daily_drift = drift / 252

        # Precompute random shocks for efficiency
        dt = 1 / 252
        random_shocks = np.random.normal(0, 1, (max_days, num_simulations))

        # Generate all price paths at once using vectorized operations
        price_paths = np.zeros((max_days + 1, num_simulations))
        price_paths[0] = current_price

        for t in range(1, max_days + 1):
            price_paths[t] = price_paths[t - 1] * np.exp(
                (daily_drift - 0.5 * daily_volatility ** 2) + daily_volatility * random_shocks[t - 1])

            # --- Stream progress during the main calculation loop ---
            if (t + 1) % (max_days // 20) == 0:  # Update progress more frequently
                progress = 40 + int(60 * (t + 1) / max_days)
                yield {"status": f"Simulating day {t + 1}/{max_days}...", "progress": progress}
                await asyncio.sleep(0.01)

        # --- Compile final results ---
        predictions = {}
        price_paths_df = pd.DataFrame(price_paths)

        for name, days in time_horizons.items():
            final_prices_at_horizon = price_paths_df.iloc[days]

            expected_price = final_prices_at_horizon.mean()
            # 90% confidence interval (5th to 95th percentile) for a robust range
            lower_bound = np.percentile(final_prices_at_horizon, 5)
            upper_bound = np.percentile(final_prices_at_horizon, 95)

            predictions[name] = {
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'expected_price': round(expected_price, 2),
                'lower_change_percent': round((lower_bound / current_price - 1) * 100, 1),
                'upper_change_percent': round((upper_bound / current_price - 1) * 100, 1)
            }

        final_result = {
            "current_price": round(current_price, 2),
            "predictions": predictions,
            "basis": "Transformer AI + Geometric Brownian Motion (5000 paths)",
            "annual_volatility": round(annual_volatility * 100, 1),
            "timestamp": datetime.now().isoformat()
        }
        yield {"status": "Done", "progress": 100, "result": final_result}

    def _calculate_model_confidence(self, market_data: pd.DataFrame,
                                    current_prediction: float) -> float:
        """Calculate confidence score"""
        if len(market_data) < 20: return 0.6
        returns = market_data.get('Returns', market_data['Close'].pct_change()).dropna()
        if len(returns) > 10:
            recent_volatility = returns[-10:].std()
            confidence = max(0.5, 0.7 - (recent_volatility * 8))
            return round(confidence, 2)
        return 0.65


# Global instance
prediction_model = RealPredictionModel()


def initialize_model():
    global prediction_model
    success = prediction_model.load_model()
    if success:
        logger.info("✅ Prediction model initialized successfully!")
    else:
        logger.error("❌ Failed to initialize prediction model!")
    return success