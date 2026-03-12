"""
Spaceship Titanic – Step 4: Evaluation
Loads the trained model from an MLflow run, evaluates on the
validation set, and logs metrics back to that same run.
"""

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def evaluate(X_val, y_val, run_id: str):
    """
    Load model from MLflow run_id, compute metrics, log them, and return them.
    Returns (accuracy, precision, recall, roc_auc).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    acc  = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="macro")
    rec  = recall_score(y_val, preds, average="macro")
    auc  = roc_auc_score(y_val, proba)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_accuracy",  acc)
        mlflow.log_metric("val_precision", prec)
        mlflow.log_metric("val_recall",    rec)
        mlflow.log_metric("val_roc_auc",   auc)

    print(
        f"Evaluation | Accuracy={acc:.4f} | Precision={prec:.4f} "
        f"| Recall={rec:.4f} | AUC={auc:.4f}"
    )
    return acc, prec, rec, auc


if __name__ == "__main__":
    from data_ingestion import ingest_data
    from pre_processing import preprocess
    from train import train

    df                                   = ingest_data()
    X_train, X_val, y_train, y_val, _    = preprocess(df)
    run_id, _                            = train(X_train, X_val, y_train, y_val)
    evaluate(X_val, y_val, run_id)
