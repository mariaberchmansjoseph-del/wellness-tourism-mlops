import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

def load_artifacts():
    print("── LOADING TRAIN/TEST DATA ───────────────────────────")
    train_df = pd.read_csv("data/train.csv")
    test_df  = pd.read_csv("data/test.csv")
    X_train  = train_df.drop("ProdTaken", axis=1)
    y_train  = train_df["ProdTaken"]
    X_test   = test_df.drop("ProdTaken",  axis=1)
    y_test   = test_df["ProdTaken"]
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_models():
    return {
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth":         [3, 5, 10],
                "min_samples_split": [2, 5],
                "criterion":         ["gini", "entropy"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth":    [3, 5],
                "max_features": ["sqrt", "log2"]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5]
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.5, 1.0]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            ),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5]
            }
        },
        "Bagging": {
            "model": BaggingClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [10, 20],
                "max_samples":  [0.8, 1.0]
            }
        }
    }

def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4)
    }

def train_all_models(X_train, X_test, y_train, y_test):
    print("\n── MODEL TRAINING WITH MLFLOW TRACKING ──────────────")

    # Set MLflow tracking URI BEFORE any logging calls
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wellness-tourism-prediction")

    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models     = get_models()
    results    = {}
    best_score = -1
    best_model = None
    best_name  = None

    for model_name, config in models.items():
        print(f"\n  Training {model_name}...")
        with mlflow.start_run(run_name=model_name):

            # Hyperparameter tuning
            gs = GridSearchCV(
                estimator  = config["model"],
                param_grid = config["params"],
                cv         = cv,
                scoring    = "roc_auc",
                n_jobs     = -1
            )
            gs.fit(X_train, y_train)

            best_est    = gs.best_estimator_
            best_params = gs.best_params_
            cv_score    = gs.best_score_

            # Log parameters to MLflow
            mlflow.log_param("model_name", model_name)
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            # Evaluate and log metrics
            metrics = evaluate_model(best_est, X_test, y_test)
            mlflow.log_metric("cv_roc_auc",      round(cv_score, 4))
            mlflow.log_metric("test_accuracy",   metrics["accuracy"])
            mlflow.log_metric("test_f1",         metrics["f1_score"])
            mlflow.log_metric("test_precision",  metrics["precision"])
            mlflow.log_metric("test_recall",     metrics["recall"])
            mlflow.log_metric("test_roc_auc",    metrics["roc_auc"])

            # Log model artifact
            mlflow.sklearn.log_model(best_est, "model")

            print(f"    Best params: {best_params}")
            print(f"    CV AUC:      {cv_score:.4f}")
            print(f"    Test AUC:    {metrics['roc_auc']:.4f}")
            print(f"    Accuracy:    {metrics['accuracy']:.4f}")

            results[model_name] = {
                "model":    best_est,
                "params":   best_params,
                "cv_score": cv_score,
                "metrics":  metrics
            }

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = best_est
                best_name  = model_name

    return results, best_model, best_name, best_score

def print_comparison(results):
    print("\n── MODEL COMPARISON ──────────────────────────────────")
    print(f"  {'Model':20} {'AUC':10} {'Accuracy':10} {'F1':10}")
    print("  " + "-" * 50)
    sorted_r = sorted(results.items(),
                      key=lambda x: x[1]["metrics"]["roc_auc"],
                      reverse=True)
    for name, r in sorted_r:
        m = r["metrics"]
        print(f"  {name:20} {m['roc_auc']:10.4f} "
              f"{m['accuracy']:10.4f} {m['f1_score']:10.4f}")

def save_best_model(model, model_name, metrics):
    print("\n── SAVING BEST MODEL ─────────────────────────────────")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    metadata = {
        "model_name": model_name,
        "metrics":    metrics,
        "features":   list(model.feature_names_in_)
                      if hasattr(model, "feature_names_in_") else []
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Best model ({model_name}) saved to models/best_model.pkl")
    print(f"  Metadata saved to models/model_metadata.json")

def main():
    print("=" * 60)
    print("MODEL TRAINING WITH EXPERIMENTATION TRACKING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_artifacts()
    results, best_model, best_name, best_score = train_all_models(
        X_train, X_test, y_train, y_test
    )
    print_comparison(results)
    best_metrics = results[best_name]["metrics"]
    save_best_model(best_model, best_name, best_metrics)

    print(f"\n{'='*60}")
    print(f"✓ TRAINING COMPLETE")
    print(f"  Best Model: {best_name}")
    print(f"  Best AUC:   {best_score:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
