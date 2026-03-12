"""
Spaceship Titanic – Step 3: Training
Trains a Logistic Regression classifier with Optuna hyperparameter
optimization and logs every trial + final model to MLflow.
Saves the optimised model to artifacts/model.pkl.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import optuna

from pathlib import Path
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACTS_DIR        = Path("artifacts")
MLFLOW_TRACKING_URI  = "sqlite:///mlflow.db"
EXPERIMENT_NAME      = "spaceship_titanic"
RANDOM_STATE         = 42
N_TRIALS_LR          = 30
N_SPLITS             = 5


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _make_optuna_callback(parent_run_id: str):
    """Log each Optuna trial as a nested MLflow child run."""
    def callback(study, trial):
        with mlflow.start_run(
            run_name=f"LR_trial_{trial.number:03d}",
            nested=True,
            tags={"mlflow.parentRunId": parent_run_id},
        ):
            mlflow.set_tag("stage",        "optuna_trial")
            mlflow.set_tag("model_type",   "LogisticRegression")
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.set_tag("trial_state",  str(trial.state))
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_accuracy", trial.value)
            mlflow.log_metric("is_best", 1 if trial.value == study.best_value else 0)
    return callback


# ── Optuna objective ──────────────────────────────────────────────────────────

def _build_objective(X_train, y_train, cv):
    """Return an Optuna objective closure over the training data."""
    def objective(trial):
        params = {
            "C"           : trial.suggest_float("C", 0.001, 100, log=True),
            "penalty"     : trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver"      : trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "max_iter"    : trial.suggest_int("max_iter", 100, 2000),
            "random_state": RANDOM_STATE,
        }
        model  = LogisticRegression(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()
    return objective


# ── Main train entry point ────────────────────────────────────────────────────

def train(X_train, X_val, y_train, y_val):
    """
    Run Optuna HPO, retrain best LR on full train split,
    log to MLflow, and save pickle.
    Returns run_id of the final MLflow run.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    study = optuna.create_study(
        direction  = "maximize",
        study_name = "logreg_optuna",
        sampler    = TPESampler(seed=RANDOM_STATE),
    )

    print(f"Optimizing Logistic Regression ({N_TRIALS_LR} trials)...")

    with mlflow.start_run(run_name="LogReg_Optuna") as parent_run:
        mlflow.set_tag("stage",        "optuna_parent")
        mlflow.set_tag("model_type",   "LogisticRegression")
        mlflow.set_tag("n_trials",     str(N_TRIALS_LR))
        mlflow.set_tag("optimization", "optuna_tpe")

        callback = _make_optuna_callback(parent_run.info.run_id)
        study.optimize(
            _build_objective(X_train, y_train, cv),
            n_trials=N_TRIALS_LR,
            show_progress_bar=True,
            callbacks=[callback],
        )

        best_params = study.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy",   study.best_value)
        mlflow.log_metric("best_trial_number",  study.best_trial.number)
        mlflow.log_metric("n_trials_completed", len(study.trials))

    # ── Retrain best model on full train split ────────────────────────────────
    best_model = LogisticRegression(**best_params, random_state=RANDOM_STATE)
    cv_scores  = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    best_model.fit(X_train, y_train)

    print(f"Best params  : {best_params}")
    print(f"Best CV      : {study.best_value:.4f}  |  trial #{study.best_trial.number}")

    # ── Log final run ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="Final_LR_Optimized") as final_run:
        mlflow.set_tag("stage",        "final")
        mlflow.set_tag("model_type",   "LogisticRegression")
        mlflow.set_tag("optimization", "optuna_tpe")

        mlflow.log_params(best_params)

        for fold_i, score in enumerate(cv_scores):
            mlflow.log_metric("cv_fold_accuracy", score, step=fold_i)
        mlflow.log_metric("cv_accuracy_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_accuracy_std",  float(cv_scores.std()))

        mlflow.sklearn.log_model(
            sk_model             = best_model,
            name                 = "model",
            registered_model_name= "SpaceshipTitanic_LR",
        )

        run_id = final_run.info.run_id

    # ── Save pickle ───────────────────────────────────────────────────────────
    joblib.dump(best_model, ARTIFACTS_DIR / "model.pkl")
    print(f"Model saved  → artifacts/model.pkl")
    print(f"MLflow run_id: {run_id}")

    return run_id, best_model


if __name__ == "__main__":
    from data_ingestion import ingest_data
    from pre_processing import preprocess
    df                               = ingest_data()
    X_train, X_val, y_train, y_val, _ = preprocess(df)
    train(X_train, X_val, y_train, y_val)
