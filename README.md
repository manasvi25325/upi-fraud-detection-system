# UPI Fraud Detection — ML Project

A beginner-friendly, end-to-end fraud detection project: synthetic UPI transaction
generator → model training/comparison → interactive Streamlit demo.

> Real UPI transaction data isn't publicly available, so this project uses a
> **synthetic data generator** that mimics realistic UPI behavior and injects
> known fraud patterns (high-value transfers to new payees, odd-hour bursts,
> mule-account fan-in, device takeover signatures).

## Project Structure
```
upi_fraud_detection/
├── generate_data.py    # Creates synthetic UPI transactions with fraud labels
├── train.py             # Trains & compares Logistic Regression / Random Forest / XGBoost
├── app.py                # Streamlit app for interactive fraud scoring
├── data/                 # Generated dataset + evaluation plots
└── models/               # Saved trained model (fraud_model.pkl)
```

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib joblib streamlit
pip install xgboost   # optional, improves results further
```

## Run it

```bash
# 1. Generate the dataset (50,000 transactions, ~2% fraud)
python generate_data.py

# 2. Train and compare models
python train.py

# 3. Launch the interactive demo
streamlit run app.py
```

## What's inside

**Features used:**
- `amount`, `hour_of_day`, `txn_type`, `sender_bank`
- `is_new_payee` — first-time payment to this VPA
- `txns_last_24h` — velocity feature
- `device_change` — device/SIM change flag
- `sender_account_age_days`, `receiver_vpa_age_days`
- `amount_to_avg_ratio` — how unusual this amount is vs. the sender's history

**Models compared:** Logistic Regression (baseline, interpretable) → Random Forest →
XGBoost (if installed). Each model is evaluated with **Precision-Recall AUC**, not
accuracy — with ~2% fraud, a model that predicts "never fraud" would still be 98%
"accurate," which is why PR-AUC and recall-at-precision are the metrics that matter.

**Class imbalance handling:** `class_weight="balanced"` for Logistic Regression and
Random Forest; `scale_pos_weight` for XGBoost.

## A note on results
On this synthetic data, models can score *very* high (even near-perfect). That's
because the injected fraud patterns are clean and rule-like by construction — in
real-world data, fraud patterns are noisier, evolve over time, and overlap more with
legitimate behavior. Treat the near-perfect score as a sign the pipeline works
correctly, not as a benchmark for real-world performance.

## Ideas to extend this project
1. **Make fraud harder to detect** — add noise/overlap between fraud and legit
   patterns in `generate_data.py` so models have to work harder (more realistic).
2. **Add a rules engine** — write simple if/else fraud rules and compare them
   against the ML model's precision/recall.
3. **Graph-based mule detection** — use `networkx` to find VPAs with abnormal
   fan-in (many senders → one receiver in a short window).
4. **Unsupervised anomaly detection** — try `IsolationForest` assuming no fraud
   labels exist at all, and see how close it gets to the supervised model.
5. **SHAP explainability** — add SHAP values to the Streamlit app so each
   prediction shows *why* it was flagged.
6. **Deploy** — push to Streamlit Community Cloud or Hugging Face Spaces for a
   live demo link to share.

## Disclaimer
This is an educational project using synthetic data. It is **not** a production
fraud detection system and should not be used to make real financial decisions.
