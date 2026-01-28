from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import sys

# Setup paths to import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ingestion_pipeline.ingestion import clean_and_normalize, feature_engineering

app = FastAPI(title="AML Fraud Detection API")

# Load Artifacts
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgb_classifier.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "robust_scaler.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(FEATURES_PATH)

class Transaction(BaseModel):
    from_bank: int
    to_bank: int
    payment_format: str
    receiving_currency: str
    payment_currency: str
    amount_received: float
    amount_paid: float

@app.post("/predict")
async def predict_fraud(data: Transaction):
    # 1. Convert input to DataFrame
    input_dict = data.dict()
    input_dict['timestamp'] = datetime.now() # Auto-generate timestamp
    df = pd.DataFrame([input_dict])

    # 2. Preprocess using modular logic
    df = clean_and_normalize(df)
    df = feature_engineering(df)

    # 3. Frequency encoding placeholder (on-the-fly for single row)
    # Ideally, you'd load the freq_map joblib here
    for col in ['from_bank', 'to_bank']:
        df[col] = 1.0 # Default frequency for single-row inference if map not provided

    # 4. Scaling
    amt_cols = ['amount_received', 'amount_paid']
    df[amt_cols] = scaler.transform(df[amt_cols])

    # 5. Alignment
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    # 6. Predict
    prob = float(model.predict_proba(df)[:, 1][0])
    prediction = int(model.predict(df)[0])

    return {
        "is_fraud": bool(prediction),
        "probability": round(prob, 4),
        "status": "High Risk" if prob > 0.5 else "Low Risk"
    }