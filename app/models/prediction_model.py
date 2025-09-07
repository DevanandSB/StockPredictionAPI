import asyncio
import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
import torch
from arch import arch_model
from plotly.subplots import make_subplots

# Suppress verbose warnings
warnings.filterwarnings('ignore')
logging.getLogger('arch').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
    def __init__(self, feature_size, d_model=128, nhead=4, num_encoder_layers=3, dropout=0.1):
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
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

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

    def load_model(self) -> bool:
        try:
            if not all(os.path.exists(path) for path in [self.model_path, self.metadata_path, self.scaler_path]):
                logger.error("Model files not found")
                return False

            # Load metadata and scaler
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)

            # Handle scaler format
            if isinstance(scaler_data, dict):
                if 'feature_scaler' in scaler_data:
                    self.scaler = scaler_data['feature_scaler']
                elif 'scaler' in scaler_data:
                    self.scaler = scaler_data['scaler']
                else:
                    # Find the first scaler object in the dictionary
                    for key, value in scaler_data.items():
                        if hasattr(value, 'transform') and hasattr(value, 'fit'):
                            self.scaler = value
                            break
            else:
                self.scaler = scaler_data

            self.feature_columns = self.metadata.get('feature_columns', [])

            # Get model configuration from metadata
            model_config = self.metadata.get('model_config', {})
            d_model = model_config.get('hidden_dim', 128)
            nhead = model_config.get('num_heads', 4)
            num_encoder_layers = model_config.get('num_layers', 3)
            dropout = model_config.get('dropout', 0.1)

            # Initialize model
            self.model = StockPredictionTransformer(
                feature_size=len(self.feature_columns),
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout
            )

            # Load weights
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True

            logger.info("✅ Stock prediction model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False

    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        try:
            # Basic price features
            df['Returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # Moving averages
            for window in [5, 20, 50, 100, 200]:
                df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

            # Volatility measures
            df['volatility_20'] = df['Returns'].rolling(window=20).std()

            # Technical indicators
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])

            # Volume analysis
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20'].replace(0, 1)

            # Date features
            df['date'] = df.index.strftime('%Y-%m-%d')
            df['day_of_week'] = df.index.dayofweek

            return df.ffill().bfill().fillna(0)

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
                    'Sector': company_data.get('sector', 'Diversified'),
                    'Volume_fund': company_data.get('volume', 1000000),
                    'avg_sentiment': company_data.get('news_sentiment', 0.5),
                    'news_count': len(company_data.get('articles', [])),
                    'sentiment_std': company_data.get('sentiment_std', 0.2),
                    'has_news': 1 if company_data.get('articles') else 0,
                    'Dividends': company_data.get('dividends', 0),
                    'Stock Splits': company_data.get('stock_splits', 0),
                    'target_return': 0.0
                }
            else:
                fundamentals = {
                    'Date_fund': datetime.now().strftime('%Y-%m-%d'),
                    'Market Cap': 1000000,
                    'Current Price': current_price,
                    'High_fund': current_price * 1.1,
                    'Low_fund': current_price * 0.9,
                    'Stock P/E': 25.0,
                    'Book Value': current_price / 3.0,
                    'Dividend Yield': 2.5,
                    'ROE': 15.0,
                    'EPS': current_price / 25.0,
                    'Debt to equity': 0.5,
                    'Price to book value': 3.0,
                    'Sector': 'Diversified',
                    'Volume_fund': 1000000,
                    'avg_sentiment': 0.5,
                    'news_count': 5,
                    'sentiment_std': 0.2,
                    'has_news': 1,
                    'Dividends': 0,
                    'Stock Splits': 0,
                    'target_return': 0.0
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

        latest_data = market_data.iloc[-1:].copy()
        fundamental_data = self._prepare_fundamental_features(symbol, current_price, company_data)

        # Merge market and fundamental data
        for col in fundamental_data.columns:
            if col in self.feature_columns:
                latest_data[col] = fundamental_data[col].iloc[0]

        latest_data['Current Price'] = current_price
        latest_data['Close'] = current_price

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in latest_data.columns:
                latest_data[col] = 0.0

        input_features = latest_data[self.feature_columns].fillna(0.0)
        processed_features = self._preprocess_features(input_features)
        scaled_features = self.scaler.transform(processed_features)

        return torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1)

    def _calculate_realistic_volatility(self, market_data: pd.DataFrame) -> float:
        """
        Calculate realistic annualized volatility using a GARCH(1,1) model.
        This adapts to different stock types automatically.
        """
        returns = market_data['Returns'].dropna() * 100  # GARCH works best with returns in %

        if len(returns) < 100:
            # Fallback to simple standard deviation if not enough data for GARCH
            return returns.std() * np.sqrt(252) / 100

        try:
            # Use a GARCH(1,1) model, which is a standard for financial time series
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off')  # disp='off' suppresses verbose output

            # Forecast the conditional variance for the next day
            forecast = res.forecast(horizon=1)
            # The forecasted variance is annualized and converted to standard deviation
            annualized_vol = np.sqrt(forecast.variance.iloc[-1, 0]) * np.sqrt(252)

            # Convert back from % to decimal and ensure it's within a very wide, reasonable range
            final_vol = annualized_vol / 100
            return max(min(final_vol, 2.0), 0.10)  # Cap volatility between 10% and 200%

        except Exception as e:
            logger.warning(f"GARCH model failed, falling back to standard deviation. Error: {e}")
            # Fallback if GARCH fails for any reason
            return returns.std() * np.sqrt(252) / 100

    def _calculate_realistic_bounds(self, current_price: float, expected_return: float,
                                    annual_volatility: float, days: int) -> Tuple[float, float]:
        """Calculate realistic bounds using analytical solution based on dynamic volatility."""
        # Convert to daily values
        daily_return = expected_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)

        # Calculate confidence intervals using log-normal distribution properties
        # For 68% confidence interval (1 standard deviation)
        log_mean = np.log(current_price) + (daily_return - 0.5 * daily_volatility ** 2) * days
        log_std = daily_volatility * np.sqrt(days)

        lower_bound = np.exp(log_mean - log_std)
        upper_bound = np.exp(log_mean + log_std)

        # Calculate expected price
        expected_price = current_price * np.exp(daily_return * days)

        # Ensure bounds are realistic and ordered
        lower_bound = min(lower_bound, expected_price)
        upper_bound = max(upper_bound, expected_price)

        return lower_bound, upper_bound

    def create_stock_chart(self, market_data: pd.DataFrame, predictions: Dict[str, Any]) -> str:
        """Create a modern, interactive stock chart with technical indicators"""
        try:
            # Create subplots with 2 rows
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price with Moving Averages', 'Volume'),
                row_width=[0.7, 0.3]
            )

            # Add price data
            fig.add_trace(
                go.Candlestick(
                    x=market_data.index,
                    open=market_data['Open'],
                    high=market_data['High'],
                    low=market_data['Low'],
                    close=market_data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Add moving averages
            for ma in [50, 200]:
                if f'MA_{ma}' in market_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=market_data.index,
                            y=market_data[f'MA_{ma}'],
                            name=f'{ma}MA',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )

            # Add volume
            colors = ['red' if market_data['Close'].iloc[i] < market_data['Open'].iloc[i] else 'green'
                      for i in range(len(market_data))]

            fig.add_trace(
                go.Bar(
                    x=market_data.index,
                    y=market_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

            # Add RSI if available
            if 'RSI' in market_data.columns:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(
                    go.Scatter(
                        x=market_data.index,
                        y=market_data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    )
                )
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.update_layout(height=300, title_text="RSI Indicator")

            # Update layout
            fig.update_layout(
                title='Stock Analysis with Technical Indicators',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_white',
                height=800,
                showlegend=True
            )

            # Convert to HTML
            chart_html = fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True})

            return chart_html

        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return "<div>Error creating chart</div>"

    async def predict_stream(self, company_data: Dict, market_data: pd.DataFrame) -> AsyncGenerator[Dict, Any]:
        if not self.is_loaded:
            yield {"error": "Model not loaded.", "progress": 0}
            return

        try:
            symbol_str = company_data.get('symbol')
            current_price = company_data.get('current_price')

            if market_data.empty:
                yield {"error": f"Market data for {symbol_str} is empty."}
                return

            yield {"status": "Preparing features...", "progress": 25}
            await asyncio.sleep(0.1) # Give frontend time to update
            input_tensor = self._prepare_input_features(market_data, current_price, symbol_str, company_data)

            yield {"status": "Running AI prediction...", "progress": 40}
            await asyncio.sleep(0.5)

            with torch.no_grad():
                pred_return = self.model(input_tensor).item()
                pred_return = max(min(pred_return, 1.0), -0.70)

            yield {"status": "Analyzing volatility...", "progress": 50}
            annual_volatility = self._calculate_realistic_volatility(market_data)

            time_horizons = {
                "Next Day": 1, "Next Week": 5, "Next 15 Days": 15, "Next 30 Days": 30,
                "Next 3 Months": 63, "Next 6 Months": 126, "Next Year": 252, "Next 2 Years": 504
            }
            predictions_result = {}
            total_horizons = len(time_horizons)
            completed = 0

            for name, days in time_horizons.items():
                yield {"status": f"Calculating {name} prediction...",
                       "progress": 50 + int(40 * completed / total_horizons)}

                lower_bound, upper_bound = self._calculate_realistic_bounds(
                    current_price, pred_return, annual_volatility, days)
                expected_price = current_price * (1 + pred_return * (days / 252))

                predictions_result[name] = {
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'expected_price': round(expected_price, 2),
                    'lower_change_percent': round((lower_bound / current_price - 1) * 100, 1),
                    'upper_change_percent': round((upper_bound / current_price - 1) * 100, 1)
                }
                completed += 1
                await asyncio.sleep(0.1)

            yield {"status": "Generating interactive chart...", "progress": 90}
            chart_html = self.create_stock_chart(market_data, predictions_result)

            yield {"status": "Finalizing predictions...", "progress": 95}
            final_result = {
                "current_price": round(current_price, 2),
                "predictions": predictions_result,
                "chart_html": chart_html,
                "basis": "GARCH Volatility & Analytical Bounds",
                "annual_volatility": round(annual_volatility * 100, 1),
                "expected_annual_return": round(pred_return * 100, 1),
                "timestamp": datetime.now().isoformat(),
                "model_type": "General Purpose"
            }

            yield {"status": "Done", "progress": 100, "result": final_result}

        except Exception as e:
            logger.error(f"Error in prediction stream: {e}", exc_info=True)
            yield {"error": f"Prediction failed: {str(e)}"}


# Global instance and initializer
prediction_model = RealPredictionModel()


def initialize_model():
    global prediction_model
    return prediction_model.load_model()