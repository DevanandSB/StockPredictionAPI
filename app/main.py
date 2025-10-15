import asyncio
import json
import logging
import os
from typing import Dict, List

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

# --- IMPORT THE REAL MODEL ---
from app.models.prediction_model import initialize_model, prediction_model
from app.models.schemas import HealthCheck
from app.services import data_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- APP INITIALIZATION & MIDDLEWARE ---
app = FastAPI(title="Stock Market Prediction API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES & TEMPLATES ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- GLOBAL CACHE ---
# This list will be populated on startup to avoid repeated file reads.
COMPANY_LIST: List[Dict[str, str]] = []


# --- HELPER FUNCTIONS ---
def _load_companies_to_cache():
    """
    Loads the list of supported companies into the global COMPANY_LIST cache.
    This function is called once at startup.
    """
    global COMPANY_LIST
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            COMPANY_LIST = json.load(f)
        logger.info(f"Successfully loaded and cached {len(COMPANY_LIST)} companies.")
    except Exception as e:
        logger.error(f"Error loading companies to cache: {e}")
        COMPANY_LIST = []


async def get_market_data(symbol: str) -> pd.DataFrame:
    """
    Asynchronously fetches and prepares historical market data using a thread pool executor
    to avoid blocking the main asyncio event loop.
    """
    try:
        symbol_ns = symbol + '.NS' if not any(ext in symbol for ext in ['.NS', '.BO']) else symbol
        loop = asyncio.get_event_loop()
        hist = await loop.run_in_executor(None, lambda: yf.Ticker(symbol_ns).history(period="max", interval="1d"))

        if hist.empty:
            logger.error(f"No data found for {symbol_ns}")
            return pd.DataFrame()

        return prediction_model._calculate_advanced_indicators(hist)
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()


# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the prediction model and loads company data when the application starts.
    """
    logger.info("Initializing REAL prediction model...")
    if not initialize_model():
        logger.error("FATAL: Failed to initialize REAL prediction model.")

    logger.info("Loading company data into cache...")
    _load_companies_to_cache()


# --- EXCEPTION HANDLERS ---
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.exception_handler(500)
async def server_error(request: Request, exc: HTTPException):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)


# --- WEB INTERFACE ENDPOINTS ---
# Each route now has a unique function name to avoid conflicts.
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/home", response_class=HTMLResponse, tags=["Web Interface"])
async def read_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/intro", response_class=HTMLResponse, tags=["Web Interface"])
async def read_intro(request: Request):
    return templates.TemplateResponse("intro.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/mid-sem-ppt", response_class=HTMLResponse, tags=["Web Interface"])
async def read_ppt(request: Request):
    return templates.TemplateResponse("ppt.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/designs", response_class=HTMLResponse, tags=["Web Interface"])
async def read_designs(request: Request):
    return templates.TemplateResponse("working.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/features", response_class=HTMLResponse, tags=["Web Interface"])
async def read_features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/why-us", response_class=HTMLResponse, tags=["Web Interface"])
async def read_why_us(request: Request):
    return templates.TemplateResponse("why-us.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/privacy-policy", response_class=HTMLResponse, tags=["Web Interface"])
async def read_privacy_policy(request: Request):
    return templates.TemplateResponse("privacy-policy.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/terms-and-conditions", response_class=HTMLResponse, tags=["Web Interface"])
async def read_terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request, "companies": COMPANY_LIST})


@app.get("/contact-us", response_class=HTMLResponse, tags=["Web Interface"])
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "companies": COMPANY_LIST})


# --- API ENDPOINTS ---
@app.get("/api/companies", tags=["Data"])
async def get_companies():
    """Returns the cached list of companies."""
    return COMPANY_LIST


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    return HealthCheck(status="healthy", model_loaded=prediction_model.is_loaded)


@app.get("/api/company/{symbol}", tags=["Data"])
async def get_company_data(symbol: str):
    try:
        company_name = next((c["name"] for c in COMPANY_LIST if c["symbol"] == symbol), symbol)
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)
        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)
        return {
            "symbol": symbol, "company_name": company_name,
            "fundamentals": fundamentals, "technicals": technicals, "sentiment": sentiment,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        return {"symbol": symbol, "error": f"Error fetching data: {e}", "success": False}

@app.get('/sw.js', include_in_schema=False)
def serve_sw():
    # The path needs to point to the file's location inside the container
    return FileResponse('/app/sw.js')

@app.post("/api/predict/{symbol}", tags=["Prediction"])
async def predict_stock_real(symbol: str):
    logger.info(f"Received prediction request for {symbol}")
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")
    try:
        company_name = next((c["name"] for c in COMPANY_LIST if c["symbol"] == symbol), symbol)
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
        final_result = None
        async for update in prediction_model.predict_stream(all_data, market_data):
            if update.get("status") == "Done":
                final_result = update.get("result")
        if final_result:
            logger.info(f"Successfully generated prediction for {symbol}")
            return {"prediction": final_result, "symbol": symbol}
        else:
            raise HTTPException(status_code=500, detail="Prediction failed to complete.")
    except Exception as e:
        logger.error(f"Real prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict-stream/{symbol}", tags=["Prediction"])
async def predict_stock_stream(symbol: str):
    if not prediction_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    async def event_stream():
        try:
            yield f"data: {json.dumps({'status': 'Fetching all required data...', 'progress': 5})}\n\n"
            company_name = next((c["name"] for c in COMPANY_LIST if c["symbol"] == symbol), symbol)
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