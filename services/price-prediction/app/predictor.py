import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tensorflow as tf

from app.data_fetcher import FEATURE_COLS, fetch_stock_data, prepare_sequence
from app.scaler_utils import inverse_transform_price, load_scaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[3] / 'ml_models'
_models = {}


def load_all_models() -> list[str]:
    """Load all .keras models into memory at startup."""
    loaded = []
    if not MODELS_DIR.exists():
        logger.warning('ML models directory does not exist: %s', MODELS_DIR)
        return loaded

    for model_path in MODELS_DIR.glob('*.keras'):
        parts = model_path.stem.split('_', 2)
        if len(parts) < 2:
            continue

        key = model_path.stem
        try:
            _models[key] = tf.keras.models.load_model(str(model_path))
            loaded.append(key)
            logger.info('Loaded model: %s', key)
        except Exception as exc:
            logger.error('Failed to load %s: %s', model_path.name, exc)

    logger.info('Total models loaded: %d', len(loaded))
    return loaded


def get_available_symbols() -> list[str]:
    symbols = {key.split('_')[-1] for key in _models}
    return sorted(symbols)


def predict_symbol(symbol: str, days_ahead: int = 7) -> dict:
    symbol = symbol.upper()
    bilstm_key = f'bilstm_{symbol}'
    tcn_key = f'tcn_gru_{symbol}'

    has_bilstm = bilstm_key in _models
    has_tcn = tcn_key in _models
    if not has_bilstm and not has_tcn:
        raise ValueError(f'No models found for {symbol}. Available: {get_available_symbols()}')

    df = fetch_stock_data(symbol)
    current_price = float(df['close'].iloc[-1])

    scaler = load_scaler(symbol)
    X = prepare_sequence(df, scaler, FEATURE_COLS)

    predictions = {}
    if has_bilstm:
        raw_pred = _models[bilstm_key].predict(X, verbose=0)[0][0]
        predictions['bilstm'] = inverse_transform_price(scaler, raw_pred)
    if has_tcn:
        raw_pred = _models[tcn_key].predict(X, verbose=0)[0][0]
        predictions['tcn_gru'] = inverse_transform_price(scaler, raw_pred)

    if has_bilstm and has_tcn:
        ensemble_price = (predictions['bilstm'] * 0.45 + predictions['tcn_gru'] * 0.55)
    else:
        ensemble_price = list(predictions.values())[0]

    forecast = _build_multi_day_forecast(df, scaler, symbol, has_bilstm, has_tcn, days_ahead)

    if has_bilstm and has_tcn:
        agreement = 1 - abs(predictions['bilstm'] - predictions['tcn_gru']) / current_price
        confidence = round(min(max(agreement * 100, 50), 98), 1)
    else:
        confidence = 72.0

    price_change = ensemble_price - current_price
    price_change_pct = (price_change / current_price) * 100
    direction = 'BULLISH' if price_change > 0 else 'BEARISH'

    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predicted_price': round(ensemble_price, 2),
        'price_change': round(price_change, 2),
        'price_change_pct': round(price_change_pct, 2),
        'direction': direction,
        'confidence': confidence,
        'forecast': forecast,
        'models_used': list(predictions.keys()),
        'individual_predictions': {k: round(v, 2) for k, v in predictions.items()},
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data_points_used': len(df),
    }


def _build_multi_day_forecast(df, scaler, symbol, has_bilstm, has_tcn, days=7):
    forecast = []
    current_df = df.copy()

    for day_index in range(1, days + 1):
        try:
            X = prepare_sequence(current_df, scaler, FEATURE_COLS)
            preds = []
            if has_bilstm:
                raw = _models[f'bilstm_{symbol}'].predict(X, verbose=0)[0][0]
                preds.append(inverse_transform_price(scaler, raw))
            if has_tcn:
                raw = _models[f'tcn_gru_{symbol}'].predict(X, verbose=0)[0][0]
                preds.append(inverse_transform_price(scaler, raw))

            day_price = sum(preds) / len(preds)
            forecast_date = datetime.utcnow() + timedelta(days=day_index)
            while forecast_date.weekday() >= 5:
                forecast_date += timedelta(days=1)

            forecast.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'price': round(day_price, 2),
                'day': day_index,
            })

            last_row = current_df.iloc[-1].copy()
            last_row['close'] = day_price
            last_row_df = pd.DataFrame([last_row])
            current_df = pd.concat([current_df, last_row_df], ignore_index=True)
            current_df = _recalculate_indicators(current_df)

        except Exception as exc:
            logger.warning('Day %s forecast failed: %s', day_index, exc)
            break

    return forecast


def _recalculate_indicators(df):
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, SMAIndicator, EMAIndicator
    from ta.volatility import BollingerBands

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
    df['volatility'] = df['returns'].rolling(20).std()
    df = df.dropna()
    return df
