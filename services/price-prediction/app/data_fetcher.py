import logging

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60  # must match training sequence length

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'bb_mid',
    'sma_20', 'ema_12', 'returns', 'volatility'
]


def fetch_stock_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    """Fetch OHLCV + technical indicators for a symbol."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df.columns = [c.lower() for c in df.columns]
    df = df[[col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]]
    df = df.dropna()

    if df.empty:
        raise ValueError(f"Not enough OHLCV data for {symbol}")

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()

    df = df.dropna()
    if df.empty:
        raise ValueError(f"Not enough computed data for {symbol}")

    return df


def prepare_sequence(df: pd.DataFrame, scaler, feature_cols: list) -> np.ndarray:
    """Scale and create sequence for model input."""
    features = df[feature_cols].values
    scaled = scaler.transform(features)

    sequence = scaled[-SEQUENCE_LENGTH:]
    if len(sequence) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data: need {SEQUENCE_LENGTH}, got {len(sequence)}")

    return sequence.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
