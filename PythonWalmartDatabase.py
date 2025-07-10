# --- Imports ---
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import pandas as pd
import numpy as np
import string
import os
from rapidfuzz import process, fuzz

# --- MongoDB Connection ---
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

db = client["WalmartDatabase"]
customers = db["customers"]
orders = db["orders"]
fraudsummary = db["fraudsummary"]

# --- Return Reason Utilities ---
base_categories = {
    'EXPIRED': 0,
    'BROKEN': 0,
    'NOT AS DESCRIBED': 0,
    'DAMAGED': 0,
    'DEFECTIVE': 0,
    "DIDN'T WORK": 0,
    "DIDN'T LIKE": 1,
    "PET DIDN'T LIKE": 1,
    'WRONG BOOK': 2,
    "DIDN'T FIT": 1,
    'NO LONGER NEEDED': 3,
    'NOT NEEDED': 3,
    'ALLERGIC REACTION': 1,
    'WRONG SIZE': 1,
    'TOO SMALL': 1,
    "DIDN'T LIKE STYLE": 1
}

def normalize_reason(reason):
    if not isinstance(reason, str):
        return ""
    reason = reason.upper().replace("-", " ").replace("_", " ")
    reason = reason.translate(str.maketrans('', '', string.punctuation))
    return reason.strip()

def map_to_base(norm_reason):
    match, score, _ = process.extractOne(norm_reason, base_categories.keys(), scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "OTHER"

def normalize_category(cat):
    if not isinstance(cat, str):
        return ""
    cat = cat.upper().replace("-", " ").replace("_", " ")
    cat = cat.translate(str.maketrans('', '', string.punctuation))
    return cat.strip()

def map_to_base_category(norm_cat):
    match, score, _ = process.extractOne(norm_cat, base_product_categories, scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "OTHER"


def days_between(date1, date2):
    if pd.isnull(date1) or pd.isnull(date2):
        return np.nan
    return (date2 - date1).days

def safe_div(a, b):
    return a / b if b != 0 else 0

def compute_fraud_score(row):
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

    total_score = raw_score + prop_score
    return raw_score, prop_score

def convert_numpy_types(d):
    for k, v in d.items():
        if isinstance(v, np.bool_):
            d[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
    return d

# --- Main Calculation and Normalization ---
today = datetime.now()
fraud_docs = []

for cust in customers.find():
    custid = cust['custid']
    account_age = days_between(cust['createdDate'], today)
    cust_orders = list(orders.find({'custid': custid}))
    total_orders = len(cust_orders)
    total_returns = sum(1 for o in cust_orders if o.get('return_label'))
    aov = sum(o['transaction_value'] for o in cust_orders) / total_orders if total_orders else 0
    ret_orders = [o for o in cust_orders if o.get('return_label')]
    arv = sum(o['transaction_value'] for o in ret_orders) / len(ret_orders) if ret_orders else 0
    rwinabuse = sum(1 for o in ret_orders if days_between(o['order_date'], o['return_date']) > 65)
    rhighvalueabuse = sum(1 for o in ret_orders if o['transaction_value'] > 1.5 * aov)
    item_counts = {}
    for o in ret_orders:
        iid = o.get('return_item_id')
        if iid:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    rcycle = sum(1 for v in item_counts.values() if v > 1)
    categories = set(
    map(
        map_to_base_category,
        [normalize_category(o.get('return_category', '')) for o in ret_orders if o.get('return_category')]
    )
)
    rcategory = len(categories)
    reason_data = []
    for o in ret_orders:
        reason = o.get('return_reason', '')
        norm = normalize_reason(reason)
        base = map_to_base(norm)
        vague = base_categories.get(base, 2)
        reason_data.append({'BaseCategory': base, 'VaguenessScore': vague})
    if reason_data:
        df_reason = pd.DataFrame(reason_data)
        rvague = df_reason['VaguenessScore'].sum()
        rdiversity = df_reason['BaseCategory'].nunique()
        rconsistency = len(df_reason) - rdiversity
    else:
        rvague = rdiversity = rconsistency = 0

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
    raw_score, prop_score = compute_fraud_score(doc)
    fraud_score = max(raw_score + prop_score, 0)
    doc['RawFraudScore'] = float(fraud_score)
    doc['CustID'] = custid
    doc = convert_numpy_types(doc)
    fraud_docs.append(doc)
    # --- PRINT FRAUD SCORE CALCULATION ---
    print(f"\nCustomer: {custid}")
    print(f"  Features: {doc}")
    print(f"  Raw score:  {raw_score:.3f}")
    print(f"  Prop score: {prop_score:.3f}")
    print(f"  Total fraud score (clipped): {fraud_score:.3f}")


# --- Normalize Scores ---
min_score = 0.0
max_score = 33.884774795097876
score_range = max_score - min_score if max_score != min_score else 1.0

for doc in fraud_docs:
    norm_score = 100 * (doc['RawFraudScore'] - min_score) / score_range
    doc['FraudScore'] = float(norm_score)
    # FraudLabel intentionally omitted

# --- Upsert Normalized Scores to MongoDB ---
for doc in fraud_docs:
    doc_to_upsert = doc.copy()
    doc_to_upsert.pop('RawFraudScore', None)
    fraudsummary.update_one({'CustID': doc['CustID']}, {'$set': doc_to_upsert}, upsert=True)

# --- Export to CSV ---
df = pd.DataFrame([{k: v for k, v in doc.items() if k != 'RawFraudScore'} for doc in fraud_docs])
df.to_csv('fraudsummary.csv', index=False)
print("Exported fraudsummary.csv successfully.")
print("Fraud summary updated in MongoDB with normalized scores (no FraudLabel).")
