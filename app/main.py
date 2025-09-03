from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import json
import os
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORT THE REAL MODEL ---
from app.models.prediction_model import initialize_model, prediction_model
from app.services import data_fetcher
from app.models.schemas import HealthCheck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Market Prediction API", version="3.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_companies() -> List[Dict[str, str]]:
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        return []

# --- NEW ENDPOINT ---
@app.get("/api/companies", tags=["Data"])
async def get_companies():
    """Returns the list of companies from the JSON file."""
    return load_companies()


def load_companies() -> List[Dict[str, str]]:
    # ... (no change to this function)
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        return []


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing REAL prediction model...")
    if not initialize_model():
        logger.error("FATAL: Failed to initialize REAL prediction model.")


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def read_root(request: Request):
    # ... (no change to this function)
    return templates.TemplateResponse("index.html", {"request": request, "companies": load_companies()})


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    # ... (no change to this function)
    return HealthCheck(status="healthy", model_loaded=prediction_model.is_loaded)


@app.get("/api/company/{symbol}", tags=["Data"])
async def get_company_data(symbol: str):
    # ... (no change to this function)
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


# --- UPDATED PREDICTION ENDPOINT ---
@app.post("/api/predict/{symbol}", tags=["Prediction"])
async def predict_stock_real(symbol: str):
    """
    Fetch data and make a prediction using the REAL Transformer AI model.
    """
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    try:
        # Fetch all required data points
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)

        # Combine data into a single dictionary for the model
        all_data = {**fundamentals, **technicals}

        # Make the real prediction
        result = prediction_model.predict(all_data)

        return {
            "prediction": result,
            "symbol": symbol,
            "data_sources": {
                "fundamentals": "Screener.in",
                "technicals": "Yahoo Finance"
            }
        }
    except Exception as e:
        logger.error(f"Real prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))