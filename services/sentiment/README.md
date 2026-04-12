# AlphaLens – News Sentiment Analysis Service

A standalone **Python FastAPI** microservice that analyses financial news sentiment for any stock ticker using [mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) — a DistilRoBERTa model fine-tuned on financial news.

## Architecture

```
services/sentiment/
├── app/
│   ├── __init__.py          # package marker
│   ├── main.py              # FastAPI app, routes, CORS, lifespan
│   ├── headlines.py         # mock headline generator (swap for real API)
│   ├── analyzer.py          # batch inference + consensus aggregation
│   ├── pipeline_loader.py   # singleton HF pipeline (CPU-optimised)
│   └── schemas.py           # Pydantic response models
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create a virtual environment
cd services/sentiment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# 2. Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first start the model (~330 MB) is downloaded from Hugging Face Hub and cached locally. Subsequent starts are instant.

## API

### `GET /api/sentiment/{ticker}`

| Parameter | Type   | Description                        |
|-----------|--------|------------------------------------|
| `ticker`  | string | Stock symbol (1-10 chars), e.g. `AAPL` |

**Example request:**

```bash
curl http://localhost:8000/api/sentiment/AAPL
```

**Example response:**

```json
{
  "ticker": "AAPL",
  "consensus": "Bullish",
  "score_summary": {
    "positive": 3,
    "negative": 1,
    "neutral": 1
  },
  "analysed_headlines": [
    {
      "headline": "AAPL beats Q3 earnings estimates, stock surges 8% in after-hours trading",
      "source": "Reuters",
      "sentiment_label": "positive",
      "confidence": 0.9732
    }
  ]
}
```

### `GET /`

Health check — returns `{"status": "ok", "service": "sentiment-analysis"}`.

## Interactive Docs

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## CPU Optimisation Notes

| Technique | Detail |
|-----------|--------|
| `device=-1` | Forces CPU execution (no CUDA required) |
| `lru_cache` | Model loaded once, reused for every request |
| `batch_size=8` | Batches all 5 headlines in a single forward pass |
| `truncation=True` | Guards against inputs exceeding 512 tokens |
| Lifespan warm-up | Model is pre-loaded before the first request arrives |

## Replacing Mock Headlines

Edit `app/headlines.py` and swap `get_mock_headlines` for an API call to:

- **Finnhub** (`/company-news`) — already configured in the AlphaLens frontend
- **Alpha Vantage** News Sentiment endpoint
- **NewsAPI** or **GNews**

The function signature is:

```python
def get_headlines(ticker: str, count: int = 5) -> list[dict]:
    # must return [{"headline": "...", "source": "...", "timestamp": "..."}, ...]
```
