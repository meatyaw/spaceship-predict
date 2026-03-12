"""
Spaceship Titanic – Pipeline Runner
Orchestrates: ingest → preprocess → train → evaluate
"""

from data_ingestion import ingest_data
from pre_processing  import preprocess
from train           import train
from evaluation      import evaluate

ACCURACY_THRESHOLD = 0.80


def run_pipeline():
    print("=" * 55)
    print("STEP 1 – Data Ingestion")
    print("=" * 55)
    df = ingest_data()

    print("\n" + "=" * 55)
    print("STEP 2 – Preprocessing")
    print("=" * 55)
    X_train, X_val, y_train, y_val, feature_columns = preprocess(df)

    print("\n" + "=" * 55)
    print("STEP 3 – Training (Logistic Regression + Optuna)")
    print("=" * 55)
    run_id, model = train(X_train, X_val, y_train, y_val)

    print("\n" + "=" * 55)
    print("STEP 4 – Evaluation")
    print("=" * 55)
    accuracy, precision, recall, auc = evaluate(X_val, y_val, run_id)

    print("\n" + "=" * 55)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"✅  Model APPROVED for deployment  (accuracy={accuracy:.4f})")
    else:
        print(
            f"❌  Model REJECTED  "
            f"(accuracy={accuracy:.4f} < threshold={ACCURACY_THRESHOLD})"
        )
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
