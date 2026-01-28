"""
MLOps Pipeline: Anti-Money Laundering (AML) Data Preparation
Consolidated Script including Chunk-based Undersampling, Frequency Encoding, and Robust Scaling.
"""

import pandas as pd
import numpy as np
import math
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings

# --- Configuration & Paths ---
warnings.filterwarnings('ignore')
DATA_DIR = Path("data/raw")
RAW_DATA_PATH = DATA_DIR / "HI-Small_Trans.csv"
OUTPUT_DIR = Path("models")
SCALER_DIR = Path("models")
CHUNK_SIZE = 100_000

def load_data(path: str) -> pd.DataFrame:
    print(f"--- Loading data from {path} ---")
    return pd.read_csv(path)

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Cleaning and Normalizing Columns ---")
    df.columns = df.columns.str.replace(' ', '_').str.replace('.', '_').str.lower()
    return df  

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core transformation logic. 
    This is called by BOTH training and inference pipelines.
    """
    print("--- Engineering Features ---")

    # 1. Categorical Grouping
    print("--- Grouping low-frequency categories ---")
    df['payment_format'] = df['payment_format'].apply(
        lambda x: x if x in ['Cheque', 'Credit Card', 'ACH'] else 'Others'
    )
    
    currency_whitelist = ['US Dollar', 'Euro', 'Swiss Franc', 'Yuan']
    for col in ['receiving_currency', 'payment_currency']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in currency_whitelist else 'Others')

    # 2. One-Hot Encoding
    # Note: We don't use drop_first=True in some MLOps setups to ensure 
    # explicit alignment, but keeping your logic here.
    df = pd.get_dummies(df, columns=['payment_format', 'receiving_currency', 'payment_currency'])
    
    # 3. Timestamp Processing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
    
    # 4. Drop Raw IDs
    cols_to_drop = ['timestamp', 'account', 'account_1']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def balance_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    ONLY used during training to handle class imbalance.
    """
    print(f"--- Applying Chunk-based RandomUnderSampler (Ratio: 0.1) ---")
    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    num_chunks = math.ceil(len(df) / CHUNK_SIZE)
    balanced_chunks = []

    for i in range(num_chunks):
        chunk = df.iloc[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
        if "is_laundering" not in chunk.columns: continue
        
        X_chunk = chunk.drop("is_laundering", axis=1)
        y_chunk = chunk["is_laundering"]

        if len(y_chunk.unique()) > 1:
            X_res, y_res = rus.fit_resample(X_chunk, y_chunk)
            balanced_chunks.append(pd.concat([X_res, y_res], axis=1))
    
    df_final = pd.concat(balanced_chunks, ignore_index=True)
    return df_final

def run_full_pipeline():
    # 1. Load & Basic Clean
    df = load_data(str(RAW_DATA_PATH))
    df = clean_and_normalize(df)

    # 2. Feature Engineering (Now returns ALL rows)
    df = feature_engineering(df)

    # 3. Balancing (Only for training)
    df_balanced = balance_data_for_training(df)

    # 4. Split
    print("--- Splitting Data ---")
    X = df_balanced.drop('is_laundering', axis=1)
    y = df_balanced['is_laundering']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Frequency Encoding & Scaling
    print("--- Final Scaling and SMOTE ---")
    # Save feature names for inference alignment
    joblib.dump(X_train.columns.tolist(), OUTPUT_DIR / 'model_features.joblib')
    
    for col in ['from_bank', 'to_bank']:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_map)
        X_test[col] = X_test[col].map(X_train[col].value_counts(normalize=True)).fillna(0)

    robust_scaler = RobustScaler()
    amt_cols = ['amount_received', 'amount_paid']
    X_train[amt_cols] = robust_scaler.fit_transform(X_train[amt_cols])
    X_test[amt_cols] = robust_scaler.transform(X_test[amt_cols])

    # SMOTE
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    # 6. Save Artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train_res, y_train_res, X_test, y_test), OUTPUT_DIR / 'smote_dataset_splits.joblib')
    joblib.dump(robust_scaler, OUTPUT_DIR / 'robust_scaler.pkl')
    
    print(f"✅ Training Ingestion Complete. Features: {len(X_train.columns)}")

if __name__ == "__main__":
    run_full_pipeline()