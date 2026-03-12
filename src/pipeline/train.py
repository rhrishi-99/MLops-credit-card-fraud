import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def train_and_log(X_train, X_val, y_train, y_val):
    
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 577,  # handles class imbalance (legit:fraud ratio)
        "eval_metric": "logloss",
    }
    
    mlflow.set_experiment("credit-fraud-detection")
    
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "f1_score":        f1_score(y_val, preds),
            "roc_auc":         roc_auc_score(y_val, proba),
            "precision":       precision_score(y_val, preds),
            "recall":          recall_score(y_val, preds),
        }
        mlflow.log_metrics(metrics)
        
        # Log model artifact
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}")
        
        return run.info.run_id, model, metrics