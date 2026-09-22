import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = [
    "uses_ip_literal", "ip_is_private", "ip_is_metadata_endpoint",
    "contains_encoding_obfuscation", "redirect_count",
    "contains_credentials_in_url", "port", "scheme_gopher",
    "scheme_file", "scheme_dict", "scheme_http", "scheme_https"
]

def load_and_prepare(path):
    raw_df = pd.read_csv(path)
    df_encoded = pd.get_dummies(raw_df, columns=["scheme"], prefix="scheme")
    for col in FEATURE_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[FEATURE_COLS].astype(float)
    return raw_df, X

def load_global_model():
    with open("results/global_model_weights.json") as f:
        w = json.load(f)
    model = LogisticRegression()
    model.coef_ = np.array(w["coef"])
    model.intercept_ = np.array(w["intercept"])
    model.classes_ = np.array([0, 1])
    return model, np.array(w["scaler_mean"]), np.array(w["scaler_scale"])

if __name__ == "__main__":
    model, scaler_mean, scaler_scale = load_global_model()

    orgs = {
        "org-a": "dataset/raw/org_a_requests.csv",
        "org-b": "dataset/raw/org_b_requests.csv",
        "org-c": "dataset/raw/org_c_requests.csv"
    }

    all_rows = []
    for org_name, path in orgs.items():
        raw_df, X = load_and_prepare(path)
        X_scaled = (X.values - scaler_mean) / scaler_scale
        risk_scores = model.predict_proba(X_scaled)[:, 1]
        predicted = model.predict(X_scaled)

        out = pd.DataFrame({
            "org": org_name,
            "target_host": raw_df["target_host"],
            "scheme": raw_df["scheme"],
            "port": raw_df["port"],
            "redirect_count": raw_df["redirect_count"],
            "actual_label": raw_df["is_ssrf"].map({1: "SSRF", 0: "Benign"}),
            "predicted_label": pd.Series(predicted).map({1: "SSRF", 0: "Benign"}),
            "risk_score": risk_scores.round(4)
        })
        all_rows.append(out)

    final = pd.concat(all_rows, ignore_index=True)
    final = final.sort_values("risk_score", ascending=False).reset_index(drop=True)
    final["rank"] = final.index + 1

    final.to_csv("results/scored_requests_for_dashboard.csv", index=False)
    print(f"Exported {len(final)} scored requests to results/scored_requests_for_dashboard.csv")
    print(f"\nTop 5 highest-risk requests:")
    print(final[["rank", "org", "target_host", "predicted_label", "risk_score"]].head(5).to_string(index=False))