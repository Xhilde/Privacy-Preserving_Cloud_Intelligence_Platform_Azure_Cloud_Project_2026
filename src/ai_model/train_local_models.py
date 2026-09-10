import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import json
import os

FEATURE_COLS = [
    "uses_ip_literal", "ip_is_private", "ip_is_metadata_endpoint",
    "contains_encoding_obfuscation", "redirect_count",
    "contains_credentials_in_url", "port", "scheme_gopher",
    "scheme_file", "scheme_dict", "scheme_http", "scheme_https"
]

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["scheme"], prefix="scheme")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].astype(float)
    y = df["is_ssrf"].astype(int)
    return X, y

def train_local_model(org_name, path):
    X, y = load_and_prepare(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"[{org_name}] n_train={len(X_train)}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")

    weights = {
        "org": org_name,
        "n_samples": len(X_train),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_COLS
    }
    return weights

if __name__ == "__main__":
    orgs = {
        "org-a": "dataset/raw/org_a_requests.csv",
        "org-b": "dataset/raw/org_b_requests.csv",
        "org-c": "dataset/raw/org_c_requests.csv"
    }

    os.makedirs("results", exist_ok=True)
    all_weights = []
    for org_name, path in orgs.items():
        weights = train_local_model(org_name, path)
        all_weights.append(weights)

    with open("results/local_model_weights.json", "w") as f:
        json.dump(all_weights, f, indent=2)

    print("\nSaved local model weights to results/local_model_weights.json")