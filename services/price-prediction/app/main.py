import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from redis.exceptions import RedisError

from app.predictor import get_available_symbols, load_all_models, predict_symbol
from app.config import MAJOR_COMPANIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
REDIS_TTL = int(os.environ.get('REDIS_TTL_SECONDS', '300'))

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

async def refresh_major_companies_periodically():
    """Refresh predictions for major companies every hour."""
    while True:
        try:
            logger.info('Refreshing major companies predictions...')
            for sym in MAJOR_COMPANIES:
                try:
                    data = await asyncio.to_thread(predict_symbol, sym, 7)
                    cache_prediction(sym, data)
                except Exception as exc:
                    logger.warning('Failed to refresh %s: %s', sym, exc)
            logger.info('Major companies refreshed.')
        except Exception as exc:
            logger.error('Refresh task error: %s', exc)
        await asyncio.sleep(3600)  # 1 hour

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Loading ML models...')
    loaded = load_all_models()
    logger.info('Ready. Models loaded: %s', loaded)
    
    # Start background refresh task
    refresh_task = asyncio.create_task(refresh_major_companies_periodically())
    
    yield
    
    # Cancel task on shutdown
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title='AlphaLens ML Prediction API',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'https://localhost:3000'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def cache_prediction(symbol: str, payload: dict) -> None:
    # Redis disabled locally to prevent Error 10061 logs
    pass

def get_cached_prediction(symbol: str) -> dict | None:
    # Redis disabled locally
    return None


def generate_mock_prediction(symbol: str, days: int = 7) -> dict:
    import random
    from datetime import datetime, timedelta
    logger.warning('Using mock fallback data for %s', symbol)
    current_price = 150.0 + random.uniform(-20, 20)
    forecast = []
    last_price = current_price
    
    for day_index in range(1, days + 1):
        drift = random.uniform(-0.015, 0.02)
        day_price = last_price * (1 + drift)
        forecast_date = datetime.utcnow() + timedelta(days=day_index)
        
        while forecast_date.weekday() >= 5:
            forecast_date += timedelta(days=1)
            
        forecast.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'price': round(day_price, 2),
            'day': day_index,
        })
        last_price = day_price
        
    final_price = forecast[-1]['price']
    price_change = final_price - current_price
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predicted_price': round(final_price, 2),
        'price_change': round(price_change, 2),
        'price_change_pct': round((price_change / current_price) * 100, 2),
        'direction': 'BULLISH' if price_change > 0 else 'BEARISH',
        'confidence': round(75.5 + random.uniform(-5, 10), 1),
        'forecast': forecast,
        'models_used': ['FALLBACK_MOCK'],
        'individual_predictions': {'mock': round(final_price, 2)},
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data_points_used': 60,
        'is_fallback': True
    }


@app.get('/')
async def health() -> dict:
    return {'status': 'ok', 'models': len(get_available_symbols())}


@app.get('/models/status')
async def models_status() -> dict:
    return {
        'available_symbols': get_available_symbols(),
        'total': len(get_available_symbols()),
    }


@app.post('/predict/{symbol}')
async def predict(symbol: str, days: int = 7):
    symbol = symbol.upper()
    cached = get_cached_prediction(symbol)
    if cached:
        return cached

    try:
        result = await asyncio.to_thread(predict_symbol, symbol, days)
        cache_prediction(symbol, result)
        print(result)
        return result
    except Exception as exc:
        logger.error('Prediction error for %s: %s. Using MOCK FALLBACK.', symbol, exc)
        return generate_mock_prediction(symbol, days)



@app.get('/predict/batch')
async def predict_batch(symbols: str, days: int = 7) -> dict:
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    results = {}
    for sym in symbol_list:
        try:
            cached = get_cached_prediction(sym)
            if cached:
                results[sym] = cached
                continue
            data = await asyncio.to_thread(predict_symbol, sym, days)
            cache_prediction(sym, data)
            results[sym] = data
        except Exception as exc:
            logger.error('Prediction error for %s: %s. Using MOCK FALLBACK.', sym, exc)
            results[sym] = generate_mock_prediction(sym, days)
    return results


@app.post('/refresh/major')
async def refresh_major_companies(days: int = 7) -> dict:
    """Refresh predictions for major companies."""
    results = {}
    for sym in MAJOR_COMPANIES:
        try:
            data = await asyncio.to_thread(predict_symbol, sym, days)
            cache_prediction(sym, data)
            results[sym] = 'refreshed'
        except Exception as exc:
            results[sym] = {'error': str(exc)}
    return results


class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, symbol: str):
        await ws.accept()
        self.active.setdefault(symbol, []).append(ws)

    def disconnect(self, ws: WebSocket, symbol: str) -> None:
        if symbol in self.active:
            self.active[symbol].remove(ws)


manager = ConnectionManager()


@app.websocket('/ws/live/{symbol}')
async def websocket_live(websocket: WebSocket, symbol: str):
    await manager.connect(websocket, symbol.upper())
    try:
        while True:
            try:
                result = await asyncio.to_thread(predict_symbol, symbol.upper())
            except Exception as exc:
                logger.error('WS Prediction error for %s: %s. Using MOCK FALLBACK.', symbol, exc)
                result = generate_mock_prediction(symbol.upper(), 7)
            await websocket.send_json(result)
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol.upper())

