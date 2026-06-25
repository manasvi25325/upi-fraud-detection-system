"""
app.py
------
Interactive Streamlit demo for the UPI fraud detection model.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="UPI Fraud Detector", page_icon="🛡️")

st.title("🛡️ UPI Fraud Detection Demo")
st.write("Enter transaction details below to get a fraud risk score from the trained model.")

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

try:
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
except FileNotFoundError:
    st.error("Model not found. Run `python generate_data.py` then `python train.py` first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=500.0, step=50.0)
    hour_of_day = st.slider("Hour of Day", 0, 23, 14)
    is_new_payee = st.checkbox("New payee (never paid before)?", value=False)
    txns_last_24h = st.number_input("Transactions by sender in last 24h", min_value=0, value=2)
    device_change = st.checkbox("Device/SIM changed recently?", value=False)

with col2:
    sender_account_age_days = st.number_input("Sender account age (days)", min_value=0, value=365)
    receiver_vpa_age_days = st.number_input("Receiver VPA age (days)", min_value=0, value=200)
    avg_amount = st.number_input("Sender's typical transaction amount (₹)", min_value=1.0, value=500.0)
    txn_type = st.selectbox("Transaction Type", ["P2P", "P2M", "BillPay", "Recharge"])
    sender_bank = st.selectbox("Sender Bank", ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Yes Bank"])

amount_to_avg_ratio = round(amount / max(avg_amount, 1), 2)

# Encode categoricals the same way training did (alphabetical LabelEncoder order)
txn_types_sorted = sorted(["P2P", "P2M", "BillPay", "Recharge"])
banks_sorted = sorted(["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Yes Bank"])
txn_type_enc = txn_types_sorted.index(txn_type)
sender_bank_enc = banks_sorted.index(sender_bank)

input_df = pd.DataFrame([{
    "amount": amount,
    "hour_of_day": hour_of_day,
    "is_new_payee": int(is_new_payee),
    "txns_last_24h": txns_last_24h,
    "device_change": int(device_change),
    "sender_account_age_days": sender_account_age_days,
    "receiver_vpa_age_days": receiver_vpa_age_days,
    "amount_to_avg_ratio": amount_to_avg_ratio,
    "txn_type_enc": txn_type_enc,
    "sender_bank_enc": sender_bank_enc,
}])[features]

st.divider()

if st.button("🔍 Check Transaction", type="primary"):
    prob = model.predict_proba(input_df)[0, 1]
    pred = "FRAUD" if prob >= 0.5 else "LEGITIMATE"

    if prob >= 0.7:
        st.error(f"⚠️ High fraud risk: {prob*100:.1f}%")
    elif prob >= 0.3:
        st.warning(f"⚠️ Moderate fraud risk: {prob*100:.1f}%")
    else:
        st.success(f"✅ Low fraud risk: {prob*100:.1f}%")

    st.metric("Predicted Class", pred)
    st.metric("Fraud Probability", f"{prob*100:.2f}%")

    if hasattr(model, "feature_importances_"):
        st.subheader("What's driving this score")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(importances)

st.divider()
st.caption("Trained on synthetic data for educational purposes — not a real fraud detection system.")
