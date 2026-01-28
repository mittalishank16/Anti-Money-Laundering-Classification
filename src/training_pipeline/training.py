from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    recall_score
)
import warnings

# Configuration
warnings.filterwarnings('ignore')
INPUT_DATA = Path("models/smote_dataset_splits.joblib") 
MODEL_OUTPUT = 'models/xgb_classifier_model.joblib'

def load_data(filepath):
    """Loads the SMOTE-balanced dataset splits."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file {filepath} not found. Please run EDA/SMOTE script first.")
    
    print(f"--- Loading data from {filepath} ---")
    return joblib.load(filepath)

def train_xgboost(X_train, y_train):
    """Initializes and trains the XGBoost Classifier."""
    print("--- Training XGBoost Classifier ---")
    
    # Initializing with standard parameters for fraud detection
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Performs comprehensive evaluation on the test set."""
    print("--- Evaluating Model Performance ---")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics

def save_model(model, path):
    """Saves the trained model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"--- Model saved successfully at {path} ---")

def main():
    try:
        # 1. Load Data
        X_train, y_train, X_test, y_test = load_data(INPUT_DATA)

        # --- ADDED THIS SECTION HERE ---
        # Save the feature names to ensure the inference pipeline 
        # uses the exact same column order and drops the target.
        feature_names = X_train.columns.tolist()
        joblib.dump(feature_names, 'models/model_features.joblib')
        print("✅ Saved feature names for inference alignment.")
        # -------------------------------
        
        # 2. Train Model
        xgb_model = train_xgboost(X_train, y_train)
        
        # 3. Evaluate
        evaluate_model(xgb_model, X_test, y_test)
        
        # 4. Save
        save_model(xgb_model, MODEL_OUTPUT)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()