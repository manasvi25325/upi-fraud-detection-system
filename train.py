"""
train.py
--------
Trains and compares fraud detection models on the synthetic UPI dataset.

Run: python train.py
Output: models/fraud_model.pkl, printed evaluation report, saved plots in data/
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURES = [
    "amount", "hour_of_day", "is_new_payee", "txns_last_24h",
    "device_change", "sender_account_age_days", "receiver_vpa_age_days",
    "amount_to_avg_ratio", "txn_type_enc", "sender_bank_enc",
]


def load_and_prepare(path="data/upi_transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    le_type = LabelEncoder()
    le_bank = LabelEncoder()
    df["txn_type_enc"] = le_type.fit_transform(df["txn_type"])
    df["sender_bank_enc"] = le_bank.fit_transform(df["sender_bank"])

    return df


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"PR-AUC: {pr_auc:.4f}  |  ROC-AUC: {roc_auc:.4f}")
    return {"name": name, "model": model, "pr_auc": pr_auc, "roc_auc": roc_auc, "y_prob": y_prob}


def plot_pr_curves(results, y_test, out_path="data/pr_curves.png"):
    plt.figure(figsize=(7, 5))
    for r in results:
        precision, recall, _ = precision_recall_curve(y_test, r["y_prob"])
        plt.plot(recall, precision, label=f"{r['name']} (AP={r['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (the metric that matters for imbalanced fraud data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved PR curve comparison to {out_path}")


def plot_feature_importance(model, feature_names, out_path="data/feature_importance.png"):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    plt.title("Feature Importance (best model)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved feature importance plot to {out_path}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare()

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Fraud rate train: {y_train.mean()*100:.2f}% | test: {y_test.mean()*100:.2f}%")

    results = []

    # 1. Logistic Regression (baseline, with class weighting for imbalance)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test, y_test))

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # 3. XGBoost (if available) — typically the strongest performer
    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model("XGBoost", xgb, X_test, y_test))
    else:
        print("\n(xgboost not installed — skipping. `pip install xgboost` to include it.)")

    plot_pr_curves(results, y_test)

    best = max(results, key=lambda r: r["pr_auc"])
    print(f"\nBest model by PR-AUC: {best['name']} ({best['pr_auc']:.4f})")

    plot_feature_importance(best["model"], FEATURES)

    joblib.dump({"model": best["model"], "features": FEATURES}, "models/fraud_model.pkl")
    print("\nSaved best model to models/fraud_model.pkl")
