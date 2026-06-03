import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Hanya atur tracking URI dan experiment jika tidak sedang dalam context 'mlflow run' (CI)
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri("file://" + os.path.join(current_dir, "mlruns"))
        mlflow.set_experiment("Latihan Credit Scoring")

    # Nyalakan autolog
    mlflow.autolog()

    # Ambil file path
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(current_dir, "dataset_clean.csv")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} tidak ditemukan!")
        sys.exit(1)

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # Biarkan autolog yang menangani start_run secara otomatis agar kompatibel dengan 'mlflow run'
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Log metrik secara manual jika perlu
    accuracy = model.score(X_test, y_test)

    # Cek apakah sudah ada run aktif (dari mlflow run), jika tidak baru buat run manual
    active_run = mlflow.active_run()
    if active_run:
        mlflow.log_metric("accuracy", accuracy)
        print(f"Menggunakan run aktif: {active_run.info.run_id}")
    else:
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)

    print(f"Berhasil! Model dilatih dengan n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Accuracy: {accuracy:.4f}")
