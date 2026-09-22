import json
import numpy as np
import pandas as pd
import shap
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
    y = raw_df["is_ssrf"].astype(int)
    return raw_df, X, y

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

    # Build a shared background sample across all orgs for the explainer's baseline
    background_frames = []
    for org_name, path in orgs.items():
        _, X, _ = load_and_prepare(path)
        background_frames.append((X.values - scaler_mean) / scaler_scale)
    background = np.vstack(background_frames)
    background_sample = shap.sample(background, 100, random_state=42)

    explainer = shap.LinearExplainer(model, background_sample, feature_names=FEATURE_COLS)

    all_explanations = []
    for org_name, path in orgs.items():
        raw_df, X, y = load_and_prepare(path)
        X_scaled = (X.values - scaler_mean) / scaler_scale
        y_pred = model.predict(X_scaled)

        flagged_idx = np.where(y_pred == 1)[0][:5]  # explain first 5 flagged requests per org
        shap_values = explainer.shap_values(X_scaled[flagged_idx])

        for i, idx in enumerate(flagged_idx):
            row = raw_df.iloc[idx]
            top_features = sorted(zip(FEATURE_COLS, shap_values[i]), key=lambda t: abs(t[1]), reverse=True)[:3]

            explanation = {
                "org": org_name,
                "target_host": str(row["target_host"]),
                "scheme": row["scheme"],
                "actual_label": "SSRF" if row["is_ssrf"] == 1 else "benign",
                "top_reasons": [{"feature": f, "shap_value": round(float(v), 3)} for f, v in top_features]
            }
            all_explanations.append(explanation)

            print(f"[{org_name}] Flagged: {row['target_host']} (scheme={row['scheme']}, actual={explanation['actual_label']})")
            for f, v in top_features:
                direction = "pushed toward SSRF" if v > 0 else "pushed toward benign"
                print(f"    {f}: {v:+.3f} ({direction})")

    with open("results/shap_explanations.json", "w") as f:
        json.dump(all_explanations, f, indent=2)

    print(f"\nSaved {len(all_explanations)} explanations to results/shap_explanations.json")