import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    
    # Simulate drift: split by time (first 80% = train baseline, last 20% = "production")
    df = df.sort_values("Time")
    train_df = df.iloc[:int(len(df)*0.8)]
    prod_df   = df.iloc[int(len(df)*0.8):]
    
    # Features + label
    X = train_df.drop(columns=["Class", "Time"])
    y = train_df["Class"]
    
    # Scale Amount (V1–V28 are already PCA'd)
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df.to_parquet("data/train_baseline.parquet", index=False)

    return X_train, X_val, y_train, y_val, prod_df, scaler