from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.train import train_and_log
from src.registry.promote import register_and_promote

X_train, X_val, y_train, y_val, prod_df, scaler = load_and_preprocess("data/creditcard.csv")
run_id, model, metrics = train_and_log(X_train, X_val, y_train, y_val)
register_and_promote(run_id, metrics["f1_score"])