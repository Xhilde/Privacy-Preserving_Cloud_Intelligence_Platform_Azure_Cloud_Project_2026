import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

FEATURE_COLS = [
    "uses_ip_literal", "ip_is_private", "ip_is_metadata_endpoint",
    "contains_encoding_obfuscation", "redirect_count",
    "contains_credentials_in_url", "port", "scheme_gopher",
    "scheme_file", "scheme_dict", "scheme_http", "scheme_https"
]

NOISE_SCALES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]# sweeping from no privacy to strong privacy
N_TRIALS_PER_SCALE = 5  # average over multiple random draws since noise is random

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["scheme"], prefix="scheme")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].astype(float)
    y = df["is_ssrf"].astype(int)
    return X, y

def federated_average(all_weights, noise_scale):
    total_samples = sum(w["n_samples"] for w in all_weights)
    avg_coef = np.zeros_like(np.array(all_weights[0]["coef"]))
    avg_intercept = np.zeros_like(np.array(all_weights[0]["intercept"]))
    avg_scaler_mean = np.zeros_like(np.array(all_weights[0]["scaler_mean"]))
    avg_scaler_scale = np.zeros_like(np.array(all_weights[0]["scaler_scale"]))

    for w in all_weights:
        frac = w["n_samples"] / total_samples
        avg_coef += np.array(w["coef"]) * frac
        avg_intercept += np.array(w["intercept"]) * frac
        avg_scaler_mean += np.array(w["scaler_mean"]) * frac
        avg_scaler_scale += np.array(w["scaler_scale"]) * frac

    if noise_scale > 0:
        avg_coef = avg_coef + np.random.normal(0, noise_scale, avg_coef.shape)
        avg_intercept = avg_intercept + np.random.normal(0, noise_scale, avg_intercept.shape)

    return {"coef": avg_coef, "intercept": avg_intercept,
            "scaler_mean": avg_scaler_mean, "scaler_scale": avg_scaler_scale}

def evaluate_global_model(global_weights, path):
    X, y = load_and_prepare(path)
    X_scaled = (X.values - global_weights["scaler_mean"]) / global_weights["scaler_scale"]
    model = LogisticRegression()
    model.coef_ = global_weights["coef"]
    model.intercept_ = global_weights["intercept"]
    model.classes_ = np.array([0, 1])
    y_pred = model.predict(X_scaled)
    return (precision_score(y, y_pred, zero_division=0),
            recall_score(y, y_pred, zero_division=0),
            f1_score(y, y_pred, zero_division=0))

if __name__ == "__main__":
    with open("results/local_model_weights.json") as f:
        all_weights = json.load(f)

    print("Local model coefficient magnitude (avg abs value):",
          round(np.mean([np.abs(np.array(w["coef"])) for w in all_weights]), 3))

    orgs = {
        "org-a": "dataset/raw/org_a_requests.csv",
        "org-b": "dataset/raw/org_b_requests.csv",
        "org-c": "dataset/raw/org_c_requests.csv"
    }

    print("\n=== Privacy-Utility Tradeoff: F1-score vs. DP noise scale ===")
    print(f"{'Noise Scale':>12} | {'org-a F1':>9} | {'org-b F1':>9} | {'org-c F1':>9}")

    sweep_results = []
    for scale in NOISE_SCALES:
        trial_f1s = {"org-a": [], "org-b": [], "org-c": []}
        for _ in range(N_TRIALS_PER_SCALE if scale > 0 else 1):
            global_weights = federated_average(all_weights, scale)
            for org_name, path in orgs.items():
                _, _, f1 = evaluate_global_model(global_weights, path)
                trial_f1s[org_name].append(f1)

            sweep_results = []
    for scale in NOISE_SCALES:
        trial_recalls = {"org-a": [], "org-b": [], "org-c": []}
        trial_f1s = {"org-a": [], "org-b": [], "org-c": []}
        for _ in range(N_TRIALS_PER_SCALE if scale > 0 else 1):
            global_weights = federated_average(all_weights, scale)
            for org_name, path in orgs.items():
                _, recall, f1 = evaluate_global_model(global_weights, path)
                trial_recalls[org_name].append(recall)
                trial_f1s[org_name].append(f1)

        avg_recall = {org: np.mean(vals) for org, vals in trial_recalls.items()}
        avg_f1 = {org: np.mean(vals) for org, vals in trial_f1s.items()}
        mean_recall_all_orgs = np.mean(list(avg_recall.values()))
        print(f"{scale:>12} | recall(a/b/c)={avg_recall['org-a']:.2f}/{avg_recall['org-b']:.2f}/{avg_recall['org-c']:.2f} | mean_recall={mean_recall_all_orgs:.3f} | f1(a/b/c)={avg_f1['org-a']:.2f}/{avg_f1['org-b']:.2f}/{avg_f1['org-c']:.2f}")
        sweep_results.append({"noise_scale": scale, "mean_recall": mean_recall_all_orgs, **{f"recall_{k}": v for k, v in avg_recall.items()}, **{f"f1_{k}": v for k, v in avg_f1.items()}})

    best = max([r for r in sweep_results if r["mean_recall"] >= 0.85], key=lambda r: r["noise_scale"], default=sweep_results[0])
    print(f"\nChosen operating point: noise_scale={best['noise_scale']} (highest noise that keeps mean recall >= 0.85, actual mean_recall={best['mean_recall']:.3f})")

    with open("results/privacy_utility_tradeoff.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Save the final model at a reasonable, non-zero noise level for downstream use
        final_weights = federated_average(all_weights, noise_scale=best["noise_scale"])
    with open("results/global_model_weights.json", "w") as f:
        json.dump({
            "coef": final_weights["coef"].tolist(),
            "intercept": final_weights["intercept"].tolist(),
            "scaler_mean": final_weights["scaler_mean"].tolist(),
            "scaler_scale": final_weights["scaler_scale"].tolist(),
            "feature_names": FEATURE_COLS
        }, f, indent=2)

    print("\nSaved privacy-utility tradeoff table to results/privacy_utility_tradeoff.json")
    print(f"Saved final global model (noise_scale={best['noise_scale']}) to results/global_model_weights.json")