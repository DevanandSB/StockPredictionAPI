from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging
import json
import os
from typing import Dict, List

from app.models.prediction_model import prediction_model, initialize_model
from app.services import data_fetcher
from app.models.schemas import StockPredictionRequest, PredictionResult, HealthCheck
from app.services.news_analyzer import news_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Market Prediction API", version="2.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


def load_companies() -> List[Dict[str, str]]:
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        return []


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing prediction model...")
    if not initialize_model():
        logger.error("Failed to initialize prediction model")


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "companies": load_companies()})


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    return HealthCheck(status="healthy", model_loaded=prediction_model.is_loaded)


@app.get("/api/companies", tags=["Data"])
async def get_companies():
    return load_companies()


@app.get("/api/company/{symbol}", tags=["Data"])
async def get_company_data(symbol: str):
    try:
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)
        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)

        return {"fundamentals": fundamentals, "technicals": technicals, "sentiment": sentiment,
                "company_name": company_name}
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")


@app.post("/api/predict-horizons/{symbol}", tags=["Prediction"])
async def predict_stock_horizons_auto(symbol: str):
    """Automatically fetch data and make predictions for multiple horizons."""
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        # Fetch all data concurrently in a real-world scenario
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)
        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)

        # Combine data, removing non-numeric/unwanted keys
        all_data = {**fundamentals, **technicals, **sentiment}
        all_data.pop('articles', None)

        # Make prediction
        result = prediction_model.predict_horizons(all_data)

        return {
            "predictions": result,
            "company_name": company_name,
            "symbol": symbol,
            "data_sources": {
                "fundamentals": "Screener.in",
                "technicals": "NSE (via nsepy)",
                "sentiment": "NewsAPI (or simulated)"
            }
        }
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/{symbol}", tags=["Prediction"])
async def predict_stock_auto(symbol: str):
    """(Legacy) Automatically fetch data and make a short-term prediction."""
    # This endpoint now calls the new horizons prediction and returns the short-term result
    try:
        horizons_result = await predict_stock_horizons_auto(symbol)
        short_term = horizons_result['predictions']['next_day']

        return {
            "prediction": short_term['prediction_percent'],
            "confidence": short_term['confidence'],
            "model_type": short_term['basis'],
            **horizons_result
        }
    except Exception as e:
        # Re-raise HTTPException from the called function
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Legacy prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))