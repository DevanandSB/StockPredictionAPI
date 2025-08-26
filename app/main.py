from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import logging
import json
import os
from typing import Dict, Any, List

from app.models.prediction_model import prediction_model, initialize_model
from app.services.data_fetcher import data_fetcher
from app.models.schemas import (
    StockPredictionRequest,
    PredictionResult,
    HealthCheck,
    PredictionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Market Prediction API",
    description="API for stock market prediction using Fundamentals, Technicals and Sentimental analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this import at the top
from app.services.news_analyzer import news_analyzer


# Add these endpoints after your existing endpoints
@app.get("/api/news/{symbol}", tags=["News"])
async def get_company_news(symbol: str):
    """Get news articles for a company"""
    try:
        # Find company name
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        articles = news_analyzer.get_company_news(company_name, symbol)
        return {"articles": articles, "count": len(articles)}
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news-sentiment/{symbol}", tags=["News"])
async def get_news_sentiment(symbol: str):
    """Get sentiment analysis from news for a company"""
    try:
        # Find company name
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        sentiment_data = news_analyzer.analyze_news_sentiment(company_name, symbol)
        return sentiment_data
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Load companies list
def load_companies() -> List[Dict[str, str]]:
    try:
        companies_path = os.path.join(os.path.dirname(__file__), "data", "companies.json")
        with open(companies_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        return []


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing prediction model...")
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize prediction model")


# Root endpoint - serve the web interface
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def read_root(request: Request):
    companies = load_companies()
    return templates.TemplateResponse("index.html", {"request": request, "companies": companies})


# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    return HealthCheck(
        status="healthy",
        model_loaded=prediction_model.is_loaded if prediction_model else False,
        model_info=prediction_model.metadata if prediction_model and prediction_model.is_loaded else None
    )


# Get companies endpoint
@app.get("/api/companies", tags=["Data"])
async def get_companies():
    """Get list of available companies"""
    return load_companies()


# Fetch company data endpoint
@app.get("/api/company/{symbol}", tags=["Data"])
async def get_company_data(symbol: str):
    """Fetch data for a specific company"""
    try:
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)

        # Find company name
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)

        return {
            "fundamentals": fundamentals,
            "technicals": technicals,
            "sentiment": sentiment,
            "company_name": company_name
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


# Prediction endpoint
@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_stock(request: StockPredictionRequest):
    """Predict stock price based on input data"""
    try:
        if not prediction_model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Combine all data
        all_data = {**request.fundamental_data, **request.technical_data, **request.sentiment_data}

        # Make prediction
        result = prediction_model.predict(all_data)

        return PredictionResult(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simple prediction endpoint for automatic data
@app.post("/api/predict/{symbol}", tags=["Prediction"])
async def predict_stock_auto(symbol: str):
    """Automatically fetch data and make prediction for a symbol"""
    try:
        if not prediction_model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Fetch all data
        fundamentals = data_fetcher.get_stock_fundamentals(symbol)
        technicals = data_fetcher.get_technical_indicators(symbol)

        # Find company name
        companies = load_companies()
        company_name = next((c["name"] for c in companies if c["symbol"] == symbol), symbol)

        sentiment = data_fetcher.get_sentiment_data(symbol, company_name)

        # Combine all data
        all_data = {**fundamentals, **technicals, **sentiment}

        # Make prediction
        result = prediction_model.predict(all_data)

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_type": result["model_type"],
            "company_name": company_name,
            "symbol": symbol,
            "data_sources": {
                "fundamentals": "Yahoo Finance",
                "technicals": "Yahoo Finance",
                "sentiment": "Simulated based on price movement"
            }
        }

    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)