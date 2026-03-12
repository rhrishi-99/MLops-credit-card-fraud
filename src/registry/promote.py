import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-detector"

def register_and_promote(run_id: str, new_f1: float):
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version

    # Set as challenger first
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"v{version} → @challenger")

    # Check if a champion exists
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        prod_f1 = mlflow.get_run(champion.run_id).data.metrics["f1_score"]

        if new_f1 > prod_f1:
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            print(f"v{version} → @champion ✅ (F1 {prod_f1:.4f} → {new_f1:.4f})")
            return True
        else:
            print(f"v{version} stays @challenger (F1 {new_f1:.4f} < champion {prod_f1:.4f})")
            return False

    except mlflow.exceptions.MlflowException:
        # No champion yet — promote directly
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        print(f"v{version} → @champion (first deploy)")
        return True