import os

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = os.path.dirname(__file__)
SCALERS_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', '..', '..', 'ml_models', 'scalers'))
_scaler_cache = {}


def load_scaler(symbol: str) -> MinMaxScaler:
    """Load and cache scaler for a symbol. Creates a mock scaler if missing."""
    normalized = symbol.upper()
    if normalized in _scaler_cache:
        return _scaler_cache[normalized]

    path = os.path.join(SCALERS_DIR, f"{normalized}_scaler.pkl")
    if not os.path.exists(path):
         # Create a dummy scaler for simulation mode to prevent FileNotFoundError logs
         scaler = MinMaxScaler()
         # Fit with some dummy data to initialize n_features_in_
         dummy_data = np.random.rand(100, 15)  # 15 features matches FEATURE_COLS
         scaler.fit(dummy_data)
         _scaler_cache[normalized] = scaler
         return scaler

    scaler = joblib.load(path)
    _scaler_cache[normalized] = scaler
    return scaler


def inverse_transform_price(scaler, scaled_price: float, close_col_index: int = 3) -> float:
    """Inverse transform a single predicted close price."""
    n_features = scaler.n_features_in_
    dummy = np.zeros((1, n_features))
    dummy[0, close_col_index] = scaled_price

    inversed = scaler.inverse_transform(dummy)
    return float(inversed[0, close_col_index])
