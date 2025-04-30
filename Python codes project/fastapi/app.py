import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Aquaculture Disease Prediction API",
              description="Predicts diseases based on water quality parameters and provides recommendations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the Random Forest model instead of LightGBM
MODEL_PATH = r"C:\Users\naren\Desktop\Python codes project\fastapi\model"

try:
    # Load model artifacts
    models = {}
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    metadata = joblib.load(os.path.join(MODEL_PATH, "metadata.pkl"))
    
    # Load disease models - using Random Forest models
    target_cols = metadata["target_columns"]
    for disease in target_cols:
        model_path = os.path.join(MODEL_PATH, f"rf_model_{disease}.pkl")
        models[disease] = joblib.load(model_path)
    
    print(f"Successfully loaded Random Forest models for diseases: {', '.join(target_cols)}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise RuntimeError(f"Failed to load models: {str(e)}")

# Define parameter ranges and recommendations
PARAMETER_RANGES = {
    "DO": {"optimal": [5.0, 10.0], "unit": "mg/L"},
    "pH": {"optimal": [7.5, 8.5], "unit": ""},
    "Alkalinity": {"optimal": [100, 150], "unit": "mg/L"},
    "Hardness": {"optimal": [150, 250], "unit": "mg/L"},
    "Nitrite": {"optimal": [0, 0.25], "unit": "mg/L"},
    "H2S": {"optimal": [0, 0.01], "unit": "ppm"},
    "Salinity": {"optimal": [15, 25], "unit": "ppt"},
    "Ammonia": {"optimal": [0, 0.1], "unit": "mg/L"},
    "Temperature": {"optimal": [28, 32], "unit": "°C"}
}

RECOMMENDATIONS = {
    "DO": {
        "low": [
            "Increase aeration immediately using paddle wheels or blowers.",
            "Reduce feeding temporarily.",
            "Perform partial water exchange if needed."
        ],
        "high": [
            "Reduce the number of aerators slightly.",
            "No major action needed unless DO > 10 mg/L."
        ]
    },
    "pH": {
        "low": [
            "Apply agricultural lime (Calcium Carbonate - CaCO₃).",
            "Increase alkalinity through liming."
        ],
        "high": [
            "Apply organic acids or molasses to bring pH down.",
            "Perform partial water exchange."
        ]
    },
    "Alkalinity": {
        "low": [
            "Apply dolomite or agricultural lime to raise alkalinity.",
            "Regular liming to maintain buffer capacity."
        ],
        "high": [
            "Perform partial water exchange to lower alkalinity.",
            "Stop lime application temporarily."
        ]
    },
    "Hardness": {
        "low": [
            "Apply gypsum (calcium sulfate) to increase hardness.",
            "Supplement minerals if necessary."
        ],
        "high": [
            "Dilute pond water by adding freshwater.",
            "Limit further mineral additions."
        ]
    },
    "Nitrite": {
        "low": [
            "No action needed (safe if low)."
        ],
        "high": [
            "Apply probiotics containing nitrifying bacteria (Nitrosomonas).",
            "Increase aeration and reduce feeding.",
            "Perform partial water exchange if necessary."
        ]
    },
    "H2S": {
        "low": [
            "No action needed (safe if absent)."
        ],
        "high": [
            "Immediately increase aeration.",
            "Apply oxidizers like potassium permanganate (KMnO₄).",
            "Remove organic sludge from pond bottom."
        ]
    },
    "Salinity": {
        "low": [
            "Add brine water or sea salt gradually.",
            "Monitor shrimp for signs of stress."
        ],
        "high": [
            "Add freshwater slowly to reduce salinity.",
            "Avoid sudden large changes in salinity."
        ]
    },
    "Ammonia": {
        "low": [
            "No action needed (safe if low)."
        ],
        "high": [
            "Immediately reduce feeding.",
            "Apply ammonia binders like zeolite.",
            "Use nitrifying probiotics.",
            "Perform partial water exchange."
        ]
    },
    "Temperature": {
        "low": [
            "Reduce water exchange (to retain pond heat).",
            "Cover ponds with greenhouse covers or shades.",
            "Reduce feeding slightly."
        ],
        "high": [
            "Increase early morning water exchange to cool pond.",
            "Provide shade over pond.",
            "Feed during cooler parts of the day (early morning / evening)."
        ]
    }
}

# Define input model
class WaterQualityInput(BaseModel):
    DO: float = Field(..., description="Dissolved Oxygen (mg/L)")
    pH: float = Field(..., description="pH level")
    Alkalinity: float = Field(..., description="Alkalinity (mg/L)")
    Hardness: float = Field(..., description="Hardness (mg/L)")
    Nitrite: float = Field(..., description="Nitrite (mg/L)")
    H2S: float = Field(..., description="Hydrogen Sulfide (ppm)")
    Salinity: float = Field(..., description="Salinity (ppt)")
    Ammonia: float = Field(..., description="Ammonia (mg/L)")
    Temperature: float = Field(..., description="Water Temperature (°C)")

# Define response model with only the requested fields
class PredictionResponse(BaseModel):
    diseases: Dict[str, bool]
    disease_probabilities: Dict[str, float]
    recommendations: Dict[str, List[str]]


    

@app.post("/predict", response_model=PredictionResponse)
def predict_diseases(input_data: WaterQualityInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make predictions for each disease
        predictions = {}
        probabilities = {}
        
        for disease in target_cols:
            # Get probability prediction
            prob = models[disease].predict_proba(input_scaled)[0, 1]
            probabilities[disease] = float(prob)
            
            # Binary prediction (threshold = 0.5)
            predictions[disease] = bool(prob >= 0.5)
        
        # Generate recommendations
        recommendations = {}
        
        for param, value in input_data.dict().items():
            optimal_range = PARAMETER_RANGES[param]["optimal"]
            
            if value < optimal_range[0]:
                status = "low"
                recommendations[param] = RECOMMENDATIONS[param]["low"]
            elif value > optimal_range[1]:
                status = "high"
                recommendations[param] = RECOMMENDATIONS[param]["high"]
            else:
                status = "optimal"
                recommendations[param] = ["Parameter is within optimal range. No action needed."]
        
        return {
            "diseases": predictions,
            "disease_probabilities": probabilities,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)