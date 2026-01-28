"""
Inference pipeline for Anti-Money Laundering (AML) Classification.
- Imports transformation logic from src.ingestion.
- Applies saved RobustScaler and aligned feature schema.
- Returns fraud predictions for EVERY row in the input CSV.
"""

from __future__ import annotations
import argparse
import pandas as pd
import joblib
import sys
from pathlib import Path

# ----------------------------
# Path Resolution (Fix for ModuleNotFoundError)
# ----------------------------
# This ensures 'src' is findable regardless of how you run the script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion_pipeline.ingestion import clean_and_normalize, feature_engineering

# ----------------------------
# Model Artifact Paths
# ----------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgb_classifier.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "robust_scaler.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.joblib"



def align_features(df: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """
    Ensures the inference DataFrame has the exact same columns as the training set.
    """
    if features_path.exists():
        model_features = joblib.load(features_path)
        
        # 1. Add missing dummy columns with 0
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
                
        # 2. Select and order columns to match training EXACTLY
        # This also automatically drops 'is_laundering' because it won't be in model_features
        return df[model_features]
    
    print("⚠️ Warning: model_features.joblib not found. XGBoost may crash due to column mismatch.")
    return df

def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    scaler_path: Path | str = SCALER_PATH,
    features_path: Path | str = FEATURES_PATH
) -> pd.DataFrame:
    """
    Inference Pipeline: Raw Data -> Cleaning -> No Sampling -> Prediction
    """
    # 1. Modular Preprocessing (Logic from ingestion.py)
    # The new ingestion.py 'feature_engineering' now returns the full dataset (no sampler)
    df = input_df.copy()
    df = clean_and_normalize(df)
    df = feature_engineering(df)

    # 2. Handle Frequency Encoding (from_bank, to_bank)
    # Note: In a full MLOps setup, you'd load a saved freq_map. 
    # For now, we calculate it on the fly to avoid crashing, 
    # but using saved maps is better for production consistency.
    for col in ['from_bank', 'to_bank']:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_map)

    # 3. Scaling
    if Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        amt_cols = ['amount_received', 'amount_paid']
        existing_cols = [c for c in amt_cols if c in df.columns]
        if existing_cols:
            df[existing_cols] = scaler.transform(df[existing_cols])

    # 4. Feature Alignment (Essential for XGBoost)
    df = align_features(df, Path(features_path))

    # 5. Load Model & Predict
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    
    model = joblib.load(model_path)
    
    # 6. Build Output
    # Now len(df) matches len(input_df) exactly
    output_df = input_df.copy()
    output_df["is_fraud_prediction"] = model.predict(df)
    output_df["fraud_probability"] = model.predict_proba(df)[:, 1]

    return output_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Inference Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to raw transaction CSV")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path")
    
    args = parser.parse_args()

    # Load raw data
    raw_data = pd.read_csv(args.input)
    
    print(f"--- Starting inference on {len(raw_data)} rows ---")
    results = predict(raw_data)
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"✅ Inference complete. Results saved to {args.output}")