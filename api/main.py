from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os

# 1. Initialize the API
app = FastAPI(title="Fraud Detection API", version="1.0")

# 2. Load the Tuned Model and its decision threshold
# The threshold is chosen by src/threshold.py from out-of-fold training predictions
# (recall-first: catch ~85% of fraud with the fewest false alarms).
MODEL_PATH = "models/tuned_xgb_model.joblib"
THRESHOLD_PATH = "models/threshold.json"
for path, script in ((MODEL_PATH, "src/train.py"), (THRESHOLD_PATH, "src/threshold.py")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run {script} first.")
model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]

# 3. Define the Input Data Structure (Pydantic)
# This tells FastAPI exactly what data to expect.
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# 4. The Preprocessing Function
def preprocess_single(data: Transaction):
    """Applies the exact same math as Phase 2 to a single transaction."""
    # Create a dictionary of the raw data
    raw_dict = data.model_dump()
    
    # Apply the transformations
    raw_dict["Hour"] = int((raw_dict["Time"] / 3600) % 24)
    raw_dict["Amount"] = np.log1p(raw_dict["Amount"])
    
    # Drop the raw Time column
    del raw_dict["Time"]
    
    # Convert to a DataFrame with the correct column order
    # We MUST ensure the columns match the training data exactly
    df = pd.DataFrame([raw_dict])
    return df
    

# 5. The Endpoint
@app.post("/predict")
def predict_fraud(data: Transaction):
    try:
        # Preprocess the input
        X_input = preprocess_single(data)
        
        # Score the transaction, then apply our threshold (not the default 0.5)
        probability = float(model.predict_proba(X_input)[0][1])
        is_fraud = probability >= THRESHOLD

        # Return the result
        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(probability, 4),
            "threshold": round(THRESHOLD, 4),
            "status": "flagged" if is_fraud else "cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))