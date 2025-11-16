# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib # Used for loading the pre-trained model
import numpy as np
import logging
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
# We assume the model artifact (e.g., a simple scikit-learn model) 
# is packaged with the application in the same directory structure.
MODEL_PATH = "model/sample_model.pkl"

# Global variable for the model object
model = None

# Create the FastAPI application instance
app = FastAPI(
    title="ML Inference Service on EKS",
    description="A simple FastAPI service for machine learning predictions."
)

# Pydantic model for request body validation
# This defines the expected input structure for the API.
class PredictionRequest(BaseModel):
    # Example: a model that takes a list of 5 numerical features
    features: List[float]
    
    # Ensure the list contains exactly 5 items
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 1.2, 0.8, 3.1, 4.5]
            }
        }

# Pydantic model for response body
class PredictionResponse(BaseModel):
    prediction: float

# --- Application Startup Event ---
@app.on_event("startup")
async def load_model():
    """Loads the pre-trained model when the application starts."""
    global model
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError as e:
        # Crucial for cloud deployment: If the model isn't found, log error and fail startup
        logger.error(f"FATAL ERROR: {e}. The application cannot run without the model.")
        # In a real environment, you might stop the application here or use a placeholder
        model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

# --- Health Check Endpoint ---
# This is used by the Kubernetes liveness and readiness probes.
@app.get("/health")
async def health_check():
    """Endpoint for Kubernetes readiness and liveness checks."""
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    # If model failed to load, return 503 (Service Unavailable)
    raise HTTPException(status_code=503, detail="Model is not loaded or service is initializing.")

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Accepts feature data and returns a prediction from the ML model.
    """
    if model is None:
        logger.warning("Attempted prediction but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not yet ready or failed to load.")

    try:
        # Convert the list of features into a numpy array for the model
        features_array = np.array([request.features])

        # Perform inference
        prediction = model.predict(features_array)[0]
        
        logger.info(f"Received features: {request.features}, Predicted: {prediction}")

        return PredictionResponse(prediction=float(prediction))
        
    except Exception as e:
        logger.error(f"Prediction failed due to error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")

# --- Root Endpoint (Documentation) ---
@app.get("/")
async def root():
    """Redirects to the OpenAPI documentation."""
    return app.openapi()
