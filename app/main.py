from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import logging
import json
import os
import asyncio
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import yfinance as yf

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
    """
    Loads the list of supported companies from the companies.json file.
    """
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        return []


async def get_market_data(symbol: str) -> pd.DataFrame:
    """
    Asynchronously fetches and prepares historical market data.
    This logic is moved here from the model to centralize data fetching.
    """
    try:
        symbol_ns = symbol + '.NS' if not any(ext in symbol for ext in ['.NS', '.BO']) else symbol

        loop = asyncio.get_event_loop()
        hist = await loop.run_in_executor(None, lambda: yf.Ticker(symbol_ns).history(period="3y", interval="1d"))

        if hist.empty:
            logger.error(f"No data found for {symbol_ns}")
            return pd.DataFrame()

        # The prediction model instance has the indicator calculation logic we need
        return prediction_model._calculate_advanced_indicators(hist)

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()


@app.get("/api/companies", tags=["Data"])
async def get_companies():
    """Returns the list of companies from the JSON file."""
    return load_companies()


@app.on_event("startup")
async def startup_event():
    """
    Initializes the prediction model when the application starts.
    """
    logger.info("Initializing REAL prediction model...")
    if not initialize_model():
        logger.error("FATAL: Failed to initialize REAL prediction model.")


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def read_root(request: Request):
    """
    Serves the main HTML page for the web interface.
    """
    return templates.TemplateResponse("index.html", {"request": request, "companies": load_companies()})


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Provides a health check endpoint to verify API and model status.
    """
    return HealthCheck(status="healthy", model_loaded=prediction_model.is_loaded)


@app.get("/api/company/{symbol}", tags=["Data"])
async def get_company_data(symbol: str):
    """
    Fetches and returns comprehensive data for a given stock symbol.
    """
    try:
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        # Fetch all data components
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)
        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)

        return {
            "symbol": symbol,
            "company_name": company_name,
            "fundamentals": fundamentals,
            "technicals": technicals,
            "sentiment": sentiment,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "error": f"Error fetching data: {e}",
            "success": False
        }


@app.get("/api/predict-stream/{symbol}", tags=["Prediction"])
async def predict_stock_stream(symbol: str):
    """
    Fetches data and streams prediction progress using Server-Sent Events.
    """
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    async def event_stream():
        try:
            yield f"data: {json.dumps({'status': 'Fetching all required data...', 'progress': 5})}\n\n"

            companies = load_companies()
            company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

            # Fetch all necessary data components in one place
            fundamentals = data_fetcher.get_stock_fundamentals(symbol)
            technicals = data_fetcher.get_technical_indicators(symbol)
            sentiment = data_fetcher.get_sentiment_data(symbol, company_name)
            market_data = await get_market_data(symbol)

            if market_data.empty:
                yield f"data: {json.dumps({'error': 'Could not retrieve market history for prediction.'})}\n\n"
                return

            all_data = {
                "symbol": symbol, "company_name": company_name,
                **fundamentals, **technicals, **sentiment
            }

            logger.info(f"Starting prediction stream for {symbol}...")
            async for progress_update in prediction_model.predict_stream(all_data, market_data):
                yield f"data: {json.dumps(progress_update)}\n\n"
                await asyncio.sleep(0.02)

        except Exception as e:
            logger.error(f"Error in prediction stream for {symbol}: {e}", exc_info=True)
            error_message = json.dumps({"error": str(e)})
            yield f"data: {error_message}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/predict/{symbol}", tags=["Prediction"])
async def predict_stock_real(symbol: str):
    """
    Fetches data and makes a prediction using the REAL Transformer AI model.
    """
    logger.info(f"Received prediction request for {symbol}")
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    try:
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        # Fetch all data components
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)
        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)
        market_data = await get_market_data(symbol)

        if market_data.empty:
            raise HTTPException(status_code=500, detail="Could not retrieve market history for prediction.")

        all_data = {
            "symbol": symbol, "company_name": company_name,
            **fundamentals, **technicals, **sentiment
        }

        # Await the final result from the streaming prediction function
        final_result = None
        async for update in prediction_model.predict_stream(all_data, market_data):
            if update.get("status") == "Done":
                final_result = update.get("result")

        if final_result:
            logger.info(f"Successfully generated prediction for {symbol}")
            return {
                "prediction": final_result,
                "symbol": symbol,
                "data_sources": {
                    "fundamentals": "Screener.in",
                    "technicals": "Yahoo Finance",
                    "sentiment": "Custom News Analyzer"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Prediction failed to complete.")

    except Exception as e:
        logger.error(f"Real prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))