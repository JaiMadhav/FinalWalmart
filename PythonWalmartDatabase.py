# Import necessary libraries for database access, data processing, and system utilities
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import pandas as pd
import string
from rapidfuzz import process, fuzz
import os, math, numpy as np

# Load MongoDB connection string from environment variable
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))

# Test MongoDB connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# Connect to the required MongoDB collections
db = client["WalmartDatabase"]
customers = db["customers"]
orders = db["orders"]
fraudsummary = db["fraudsummary"]

# Define base return categories and their vagueness scores
base_categories = {
    "size issue": 1,
    "color issue": 1,
    "defective": 0,
    "late delivery": 2,
    "quality issue": 2,
    "wrong item": 0,
    "missing parts": 0,
    "fake product": 0,
    "packaging issue": 1,
    "didn't like": 3,
    "changed mind": 3,
    "duplicate order": 1,
    "price issue": 2,
    "other": 2
}

# Clean and normalize return reasons by removing punctuation and converting to lowercase
def normalize_reason(reason):
    reason = reason.lower()
    reason = reason.translate(str.maketrans('', '', string.punctuation))
    return reason.strip()

# Map normalized reason to the closest base category using fuzzy matching
def map_to_base(norm_reason):
    match, score, _ = process.extractOne(norm_reason, base_categories.keys(), scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "other"

# Calculate number of days between two dates
def days_between(date1, date2):
    return (date2 - date1).days

# Compute fraud score using a combination of raw weights and proportional risk features
import numpy as np

def safe_div(a, b):
    return a / b if b != 0 else 0

def compute_fraud_score(row):
    # Log-transform all relevant features
    log_TotalReturns     = np.log1p(row['TotalReturns'])
    log_TotalOrders      = np.log1p(row['TotalOrders'])
    log_AOV              = np.log1p(row['AOV'])
    log_ARV              = np.log1p(row['ARV'])
    log_AccountAge       = np.log1p(row['AccountAge'])
    log_Rwinabuse        = np.log1p(row['Rwinabuse'])
    log_Rhighvalueabuse  = np.log1p(row['Rhighvalueabuse'])
    log_Rcycle           = np.log1p(row['Rcycle'])
    log_Rcategory        = np.log1p(row['Rcategory'])
    log_Rvague           = np.log1p(row['Rvague'])
    log_Rdiversity       = np.log1p(row['Rdiversity'])
    log_Rconsistency     = np.log1p(row['Rconsistency'])

    # Raw score with log1p features
    raw_score = (
        1.0 * log_TotalReturns +
        -0.2 * log_TotalOrders +
        -0.001 * log_AOV +
        0.002 * log_ARV +
        -0.01 * log_AccountAge +
        1.0 * log_Rwinabuse +
        1.2 * log_Rhighvalueabuse +
        0.8 * log_Rcycle +
        0.5 * log_Rcategory +
        0.5 * log_Rvague +
        0.5 * log_Rdiversity +
        0.3 * log_Rconsistency
    )

    # Proportional score with log1p features
    prop_score = (
        2.0 * safe_div(log_TotalReturns, log_TotalOrders) +
        1.5 * safe_div(log_Rwinabuse, log_TotalReturns) +
        2.0 * safe_div(log_Rhighvalueabuse, log_TotalReturns) +
        1.0 * safe_div(log_Rcycle, log_TotalReturns) +
        1.0 * safe_div(log_Rcategory, log_TotalOrders) +
        1.0 * log_Rvague +
        1.0 * log_Rdiversity +
        1.0 * log_Rconsistency +
        1.0 * safe_div(log_ARV, log_AOV) +
        0.5 * safe_div(1, log_AccountAge)
    )

    # Final fraud score
    total_score = raw_score + prop_score
    return raw_score, prop_score

# Start fraud summary update process
today = datetime.now()

# Iterate through all customers
for cust in customers.find():
    custid = cust['custid']
    account_age = days_between(cust['createdDate'], today)

    # Retrieve orders for the customer
    cust_orders = list(orders.find({'custid': custid}))
    total_orders = len(cust_orders)
    total_returns = sum(1 for o in cust_orders if o.get('return_label'))

    # Average Order Value (AOV)
    aov = sum(o['transaction_value'] for o in cust_orders) / total_orders if total_orders else 0

    # Filter only returned orders
    ret_orders = [o for o in cust_orders if o.get('return_label')]

    # Average Return Value (ARV)
    arv = sum(o['transaction_value'] for o in ret_orders) / len(ret_orders) if ret_orders else 0

    # Abuse: returns after 75 days
    rwinabuse = sum(1 for o in ret_orders if days_between(o['order_date'], o['return_date']) > 75)

    # Abuse: high-value returns above 1.5x AOV
    rhighvalueabuse = sum(1 for o in ret_orders if o['transaction_value'] > 1.5 * aov)

    # Count repeated item returns
    item_counts = {}
    for o in ret_orders:
        iid = o.get('return_item_id')
        if iid:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    rcycle = sum(1 for v in item_counts.values() if v > 1)

    # Count of unique return categories used
    categories = set(o.get('return_category') for o in ret_orders if o.get('return_category'))
    rcategory = len(categories)

    # Analyze return reasons and vagueness
    reason_data = []
    for o in ret_orders:
        reason = o.get('return_reason', '')
        norm = normalize_reason(reason)
        base = map_to_base(norm)
        vague = base_categories[base]
        reason_data.append({'BaseCategory': base, 'VaguenessScore': vague})

    if reason_data:
        df_reason = pd.DataFrame(reason_data)
        rvague = df_reason['VaguenessScore'].sum()       # Total vagueness score
        rdiversity = df_reason['BaseCategory'].nunique() # How varied the reasons are
        rconsistency = len(df_reason) - rdiversity       # Consistency of reasons
    else:
        rvague = rdiversity = rconsistency = 0

    # Create document to be stored in fraudsummary
    doc = {
        'CustID': custid,
        'TotalOrders': int(total_orders),
        'TotalReturns': int(total_returns),
        'AOV': float(aov),
        'ARV': float(arv),
        'AccountAge': int(account_age),
        'Rwinabuse': int(rwinabuse),
        'Rhighvalueabuse': int(rhighvalueabuse),
        'Rcycle': int(rcycle),
        'Rcategory': int(rcategory),
        'Rvague': int(rvague),
        'Rdiversity': int(rdiversity),
        'Rconsistency': int(rconsistency)
    }

    # Calculate fraud score and determine fraud label (True if score ≥ 275.0)
    raw_score, prop_score = compute_fraud_score(doc)
    fraud_score = raw_score + prop_score
    fraud_score = max(fraud_score, 0)
    print(f"[{custid}] Raw Score: {raw_score:.7f}, Proportional Score: {prop_score:.7f}, Total Fraud Score: {fraud_score:.7f}")
    doc['FraudScore'] = float(fraud_score)
    doc['FraudLabel'] = fraud_score >= 31.0

    def convert_numpy_types(d):
        for k, v in d.items():
            if isinstance(v, np.bool_):
                d[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
        return d
    doc = convert_numpy_types(doc)

    # Upsert document into fraudsummary collection
    fraudsummary.update_one({'CustID': custid}, {'$set': doc}, upsert=True)

    # Export all fraudsummary entries to CSV (each loop overrides the same file)
    cursor = fraudsummary.find({}, {
        '_id': 0,
        'CustID': 1,
        'TotalOrders': 1,
        'TotalReturns': 1,
        'AOV': 1,
        'ARV': 1,
        'AccountAge': 1,
        'Rwinabuse': 1,
        'Rhighvalueabuse': 1,
        'Rcycle': 1,
        'Rcategory': 1,
        'Rvague': 1,
        'Rdiversity': 1,
        'Rconsistency': 1,
        'FraudScore': 1,
        'FraudLabel': 1
    })

    df = pd.DataFrame(list(cursor))
    df.to_csv('fraudsummary.csv', index=False)
    print("Exported fraudsummary.csv successfully.")

# Final log message
print("Fraud summary updated in MongoDB.")

# After all customer fraud scores have been calculated and upserted

# Fetch all fraud scores from the collection
all_docs = list(fraudsummary.find({}, {'CustID': 1, 'FraudScore': 1, '_id': 0}))
all_scores = [doc['FraudScore'] for doc in all_docs]
labels = [doc['FraudLabel'] for doc in all_docs]

if all_scores:
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    print("\n--- Normalized Fraud Scores (0–100 scale) ---")
    for doc in all_docs:
        norm_score = 100 * (doc['FraudScore'] - min_score) / score_range
        print(f"CustID: {doc['CustID']}, Raw Score: {doc['FraudScore']:.2f}, Normalized: {norm_score:.2f}")


from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Load your data ===
# Replace these with your actual data
# Example: fraud scores from your database and true labels


scores = np.array(all_scores)
y_true = np.array(labels)

# === 2. Find best threshold using F1-score ===
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_pr[best_f1_idx]
print(f"Best threshold by F1-score: {best_f1_threshold:.4f}")
print(f"Best F1-score: {f1_scores[best_f1_idx]:.4f}")

# === 3. Find best threshold using ROC (Youden's J statistic) ===
fpr, tpr, thresholds_roc = roc_curve(y_true, scores)
j_scores = tpr - fpr
best_j_idx = np.argmax(j_scores)
best_roc_threshold = thresholds_roc[best_j_idx]
print(f"Best threshold by ROC (Youden's J): {best_roc_threshold:.4f}")

# === 4. Plot Precision-Recall and ROC Curves ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recalls, precisions, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

plt.tight_layout()
plt.show()

# === 5. Apply the chosen threshold and evaluate ===
# You can use either best_f1_threshold or best_roc_threshold
chosen_threshold = best_f1_threshold
y_pred = (scores >= chosen_threshold).astype(int)

print("\nClassification Report (using best F1 threshold):")
print(classification_report(y_true, y_pred, target_names=['Not Fraud', 'Fraud']))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

