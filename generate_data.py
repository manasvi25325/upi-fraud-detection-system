"""
generate_data.py
-----------------
Generates a synthetic UPI transaction dataset with realistic features
and injected fraud patterns. Since real UPI data is not public, this
simulator creates plausible transaction behavior for learning purposes.

Run: python generate_data.py
Output: data/upi_transactions.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_USERS = 2000
N_TRANSACTIONS = 50000
FRAUD_RATE = 0.02  # ~2% fraud, realistic for imbalanced classification

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Yes Bank"]
TXN_TYPES = ["P2P", "P2M", "BillPay", "Recharge"]


def random_vpa(user_id):
    domains = ["@paytm", "@ybl", "@apl", "@upi", "@oksbi"]
    return f"user{user_id}{random.choice(domains)}"


def generate_users(n_users):
    """Create a base population of users with a 'home' transaction profile."""
    users = []
    for uid in range(n_users):
        avg_amount = np.random.gamma(shape=2.0, scale=300)  # typical spend pattern
        account_age_days = np.random.randint(10, 2000)
        home_bank = random.choice(BANKS)
        users.append({
            "user_id": uid,
            "vpa": random_vpa(uid),
            "avg_amount": avg_amount,
            "account_age_days": account_age_days,
            "bank": home_bank,
        })
    return pd.DataFrame(users)


def generate_transactions(users_df, n_txn, fraud_rate):
    records = []
    start_time = datetime(2025, 1, 1)
    n_fraud_target = int(n_txn * fraud_rate)
    fraud_indices = set(np.random.choice(n_txn, n_fraud_target, replace=False))

    for i in range(n_txn):
        sender = users_df.sample(1).iloc[0]
        is_fraud = i in fraud_indices

        txn_type = random.choice(TXN_TYPES)
        timestamp = start_time + timedelta(
            minutes=int(np.random.uniform(0, 60 * 24 * 180))  # spread over ~6 months
        )

        # New payee flag: most transactions are to known payees
        is_new_payee = np.random.rand() < 0.1

        # Receiver: usually a known/random user, but for fraud often a fresh-looking VPA
        receiver = users_df.sample(1).iloc[0]
        receiver_vpa = receiver["vpa"]

        # --- Legit transaction pattern ---
        amount = max(10, np.random.normal(sender["avg_amount"], sender["avg_amount"] * 0.3))
        hour = timestamp.hour
        txns_last_24h = np.random.poisson(1.5)
        device_change = np.random.rand() < 0.03
        receiver_vpa_age_days = np.random.randint(30, 1500)

        if is_fraud:
            # --- Inject fraud signatures ---
            fraud_pattern = random.choice(["high_value_new_payee", "odd_hour_burst", "mule_fanin", "device_takeover"])

            if fraud_pattern == "high_value_new_payee":
                amount = sender["avg_amount"] * np.random.uniform(5, 15)
                is_new_payee = True
                receiver_vpa_age_days = np.random.randint(0, 5)

            elif fraud_pattern == "odd_hour_burst":
                hour = random.choice([1, 2, 3, 4])
                timestamp = timestamp.replace(hour=hour)
                txns_last_24h = np.random.poisson(8) + 5
                amount = sender["avg_amount"] * np.random.uniform(2, 6)

            elif fraud_pattern == "mule_fanin":
                # Money funnels into a small set of "mule" receiver vpas
                receiver_vpa = f"mule{random.randint(1, 20)}@upi"
                amount = sender["avg_amount"] * np.random.uniform(1, 4)
                receiver_vpa_age_days = np.random.randint(0, 10)

            elif fraud_pattern == "device_takeover":
                device_change = True
                amount = sender["avg_amount"] * np.random.uniform(3, 10)
                is_new_payee = True

        records.append({
            "txn_id": f"TXN{i:07d}",
            "timestamp": timestamp,
            "sender_vpa": sender["vpa"],
            "sender_bank": sender["bank"],
            "receiver_vpa": receiver_vpa,
            "amount": round(amount, 2),
            "txn_type": txn_type,
            "hour_of_day": hour,
            "is_new_payee": int(is_new_payee),
            "txns_last_24h": txns_last_24h,
            "device_change": int(device_change),
            "sender_account_age_days": sender["account_age_days"],
            "receiver_vpa_age_days": receiver_vpa_age_days,
            "amount_to_avg_ratio": round(amount / max(sender["avg_amount"], 1), 2),
            "is_fraud": int(is_fraud),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating user base...")
    users = generate_users(N_USERS)

    print(f"Generating {N_TRANSACTIONS} transactions (~{FRAUD_RATE*100:.1f}% fraud)...")
    txns = generate_transactions(users, N_TRANSACTIONS, FRAUD_RATE)

    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns.to_csv("data/upi_transactions.csv", index=False)

    print("Done.")
    print(f"Total transactions: {len(txns)}")
    print(f"Fraud transactions: {txns['is_fraud'].sum()} ({txns['is_fraud'].mean()*100:.2f}%)")
    print("Saved to data/upi_transactions.csv")
