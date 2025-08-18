from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Exam Score Predictor API",
    description="A simple API that predicts exam scores based on hours studied using linear regression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None
model_info = {}

# Request model for prediction
class PredictionRequest(BaseModel):
    hours_studied: float = Field(
        ...,
        ge=0, 
        le=24, 
        description="Number of hours studied (0-24)",
        example=5.0
    )

# Response model for prediction
class PredictionResponse(BaseModel):
    hours_studied: float
    predicted_score: float
    model_confidence: str
    message: str

# Response model for model info
class ModelInfoResponse(BaseModel):
    accuracy: float
    slope: float
    intercept: float
    formula: str
    training_data_size: int
    test_accuracy: float

def train_model():
    """Train the linear regression model with sample data"""
    global model, model_info
    
    # Create sample data (same as in the notebook)
    data = {
        "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Exam_Score": [20, 40, 60, 65, 70, 80, 85, 90, 95, 100]
    }
    df = pd.DataFrame(data)
    
    # Prepare data
    X = df[["Hours_Studied"]]
    y = df["Exam_Score"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Store model information
    model_info = {
        "accuracy": train_score,
        "test_accuracy": test_score,
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "formula": f"Score = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}",
        "training_data_size": len(X_train)
    }
    
    return model

# Train model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    train_model()
    print("Model trained and API ready!")

@app.get("/", tags=["General"])
async def root():
    """Welcome message and API information"""
    return {
        "message": "Welcome to the Exam Score Predictor API!",
        "description": "This API predicts exam scores based on hours studied using linear regression",
        "endpoints": {
            "GET /": "This welcome message",
            "POST /predict": "Make a prediction",
            "GET /model-info": "Get model details",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        },
        "example_usage": {
            "curl": "curl -X POST 'http://localhost:8000/predict' -H 'Content-Type: application/json' -d '{\"hours_studied\": 5.0}'"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_score(request: PredictionRequest):
    """
    Predict exam score based on hours studied
    
    - **hours_studied**: Number of hours studied (0-24)
    - Returns predicted exam score as percentage
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained yet")
    
    try:
        # Make prediction
        hours = request.hours_studied
        predicted_score = model.predict([[hours]])[0]
        
        # Ensure score is within reasonable bounds
        predicted_score = max(0, min(100, predicted_score))
        
        # Generate confidence message
        if model_info["test_accuracy"] > 0.95:
            confidence = "High"
        elif model_info["test_accuracy"] > 0.8:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate encouraging message
        if predicted_score >= 90:
            message = "Excellent! You're on track for a great score!"
        elif predicted_score >= 70:
            message = "Good work! A solid score is within reach!"
        elif predicted_score >= 50:
            message = "You're getting there! Consider studying a bit more."
        else:
            message = "More study time needed for a passing grade."
        
        return PredictionResponse(
            hours_studied=hours,
            predicted_score=round(predicted_score, 2),
            model_confidence=confidence,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Information"])
async def get_model_info():
    """
    Get detailed information about the trained model
    
    Returns model accuracy, coefficients, and formula
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained yet")
    
    return ModelInfoResponse(**model_info)

@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns the status of the API and model
    """
    model_status = "ready" if model is not None else "not_trained"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0",
        "message": "API is running successfully"
    }

@app.get("/training-data", tags=["Model Information"])
async def get_training_data():
    """
    Get the training data used to build the model
    
    Returns the sample dataset
    """
    data = {
        "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Exam_Score": [20, 40, 60, 65, 70, 80, 85, 90, 95, 100]
    }
    
    return {
        "training_data": data,
        "description": "Sample data showing relationship between study hours and exam scores",
        "data_points": len(data["Hours_Studied"])
    }

# For running the API directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)