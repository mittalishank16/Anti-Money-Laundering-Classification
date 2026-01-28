import pandas as pd
import numpy as np
import joblib
import optuna
import mlflow
import mlflow.xgboost
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# --- Configuration ---
DATA_PATH = Path("models/smote_dataset_splits.joblib")
TRACKING_URI = r"file:///C:/Users/user/Desktop/ML & DL projects/Anti- Money Laundering classification/mlruns"
EXPERIMENT_NAME = "AML_XGBoost_Optuna_SMOTE"

# Load Data
print("Loading data...")
X_train, y_train, X_test, y_test = joblib.load(DATA_PATH)

# --- ADDED THIS SECTION HERE ---
        # Save the feature names to ensure the inference pipeline 
        # uses the exact same column order and drops the target.
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'models/model_features.joblib')
print("✅ Saved feature names for inference alignment.")
        # -------------------------------

def objective(trial):
    """Optuna objective function for XGBClassifier."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10), # Useful even with SMOTE
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    # Start a nested MLflow run for each trial
    with mlflow.start_run(nested=True):
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

    # We maximize F1-score because of the fraud imbalance
    return f1

def run_tuning():
    # Setup MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start Parent Run
    with mlflow.start_run(run_name="Optuna_XGB_Search"):
        print("Starting Optuna study...")
        # Direction is maximize because we return F1-score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15)

        print("\nBest parameters found:", study.best_trial.params)

        # Train Final Model with Best Params
        print("\nTraining final model with best parameters...")
        best_params = study.best_trial.params
        best_model = XGBClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # Final Evaluation
        y_pred_final = best_model.predict(X_test)
        y_prob_final = best_model.predict_proba(X_test)[:, 1]
        
        final_f1 = f1_score(y_test, y_pred_final)
        final_auc = roc_auc_score(y_test, y_prob_final)

        print("--- Final Tuned Model Performance ---")
        print(classification_report(y_test, y_pred_final))
        print(f"Final AUC: {final_auc:.4f}")

        # Log Final Artifacts
        mlflow.log_params(best_params)
        mlflow.log_metrics({"final_f1": final_f1, "final_auc": final_auc})
        mlflow.xgboost.log_model(best_model, artifact_path="model")
        
        # Save locally as well
        joblib.dump(best_model, "models/best_xgb_classifier.joblib")
        print(f"✅ Model saved to models/best_xgb_classifier.joblib and MLflow at {TRACKING_URI}")

if __name__ == "__main__":
    run_tuning()