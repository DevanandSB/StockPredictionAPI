from enum import Enum

class PredictionType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    COMBINED = "combined"


from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class PredictionType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    COMBINED = "combined"


class StockPredictionRequest(BaseModel):
    fundamental_data: Dict[str, Union[float, int]] = Field(..., description="Fundamental indicators")
    technical_data: Dict[str, Union[float, int]] = Field(..., description="Technical indicators")
    sentiment_data: Dict[str, Union[float, int]] = Field(..., description="Sentiment scores")
    prediction_type: PredictionType = Field(default=PredictionType.COMBINED, description="Type of prediction to make")

    @validator('*', pre=True)
    def convert_numpy_types(cls, v):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(v, dict):
            return {k: float(val) if isinstance(val, (np.floating, np.integer)) else val for k, val in v.items()}
        return v


class PredictionResult(BaseModel):
    prediction: float = Field(..., description="Predicted stock price or return")
    confidence: float = Field(..., description="Model confidence score")
    features_used: List[str] = Field(..., description="List of features used for prediction")
    model_type: str = Field(..., description="Type of model used for prediction")


class HealthCheck(BaseModel):
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
