import pandas as pd
#import talib
talib = None
import yfinance as yf


def prepare_training_data(symbols: list, period: str = "5y"):
    """Prepare comprehensive training data"""
    all_data = []

    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval="1d")

            # Calculate technical indicators
            hist = calculate_technical_indicators(hist)

            # Add fundamental data (simplified)
            hist['symbol'] = symbol

            all_data.append(hist)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    return pd.concat(all_data) if all_data else pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    # Price indicators
    df['RSI'] = talib.RSI(df['Close'])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])

    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    # Volatility
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

    # Trend
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    return df.dropna()