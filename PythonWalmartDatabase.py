import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import string
from rapidfuzz import process, fuzz
import os

# --- MongoDB Connection ---
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
customers = db["customers"]         # All customers collection
orders = db["orders"]               # Orders collection
fraudsummary = db["fraudsummary"]

# --- Helper Functions (as before) ---
base_categories = {
    'EXPIRED': 0, 'BROKEN': 0, 'NOT AS DESCRIBED': 0, 'DAMAGED': 0, 'DEFECTIVE': 0,
    "DIDN'T WORK": 0, "DIDN'T LIKE": 1, "PET DIDN'T LIKE": 1, 'WRONG BOOK': 2,
    "DIDN'T FIT": 1, 'NO LONGER NEEDED': 3, 'NOT NEEDED': 3, 'ALLERGIC REACTION': 1,
    'WRONG SIZE': 1, 'TOO SMALL': 1, "DIDN'T LIKE STYLE": 1
}
base_product_categories = [
    "ELECTRONICS", "CLOTHING", "BOOKS", "TOYS", "HOME", "GROCERY", "BEAUTY",
    "SPORTS", "AUTOMOTIVE", "FURNITURE", "JEWELRY", "BABY", "OFFICE",
    "PHARMACY", "AUTO", "PET", "HOME_IMPROVEMENT", "APPLIANCES", "CLEANING", 
    "COFFEE_APPLIANCES", "SEASONAL", "BEDDING", "OTHER"
]

def normalize_reason(reason):
    if not isinstance(reason, str): return ""
    reason = reason.upper().replace("-", " ").replace("_", " ")
    reason = reason.translate(str.maketrans('', '', string.punctuation))
    return reason.strip()

def map_to_base(norm_reason):
    match, score, _ = process.extractOne(norm_reason, base_categories.keys(), scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "OTHER"

def normalize_category(cat):
    if not isinstance(cat, str): return ""
    cat = cat.upper().replace("-", " ").replace("_", " ")
    cat = cat.translate(str.maketrans('', '', string.punctuation))
    return cat.strip()

def map_to_base_category(norm_cat):
    match, score, _ = process.extractOne(norm_cat, base_product_categories, scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "OTHER"

def days_between(date1, date2):
    if pd.isnull(date1) or pd.isnull(date2): return np.nan
    return (date2 - date1).days

def safe_div(a, b): return a / b if b != 0 else 0

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

    return float(max(raw_score + prop_score, 0))

# --- Load master dataset from CSV (no raw fraud score) ---
csv_path = 'fraudsummary.csv'
df_master = pd.read_csv(csv_path)

# --- Process new customers from MongoDB ---
today = datetime.now()
fraud_docs = []

for cust in customers.find({}, {'_id': 0}):
    custid = cust['custid']
    account_age = days_between(cust.get('createdDate'), today)
    cust_orders = list(orders.find({'custid': custid}))
    total_orders = len(cust_orders)
    total_returns = sum(1 for o in cust_orders if o.get('return_label'))
    aov = sum(o['transaction_value'] for o in cust_orders) / total_orders if total_orders else 0
    ret_orders = [o for o in cust_orders if o.get('return_label')]
    arv = sum(o['transaction_value'] for o in ret_orders) / len(ret_orders) if ret_orders else 0
    rwinabuse = sum(1 for o in ret_orders if days_between(o.get('order_date'), o.get('return_date')) > 65)
    rhighvalueabuse = sum(1 for o in ret_orders if o['transaction_value'] > 1.5 * aov)
    item_counts = {}
    for o in ret_orders:
        iid = o.get('return_item_id')
        if iid: item_counts[iid] = item_counts.get(iid, 0) + 1
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
    doc['RawFraudScore'] = compute_fraud_score(doc)
    fraud_docs.append(doc)

# --- Convert new customers to DataFrame ---
df_new = pd.DataFrame(fraud_docs)

# --- Compute raw fraud score for master if not present ---
feature_cols = [
    'TotalOrders', 'TotalReturns', 'AOV', 'ARV', 'AccountAge',
    'Rwinabuse', 'Rhighvalueabuse', 'Rcycle', 'Rcategory',
    'Rvague', 'Rdiversity', 'Rconsistency'
]

if 'RawFraudScore' not in df_master.columns:
    df_master['RawFraudScore'] = df_master[feature_cols].apply(lambda row: compute_fraud_score(row), axis=1)

# --- Combine all customers ---
for col in df_new.columns:
    if col not in df_master.columns:
        df_master[col] = np.nan
for col in df_master.columns:
    if col not in df_new.columns:
        df_new[col] = np.nan
df_new = df_new[df_master.columns]
df_all = pd.concat([df_master, df_new], ignore_index=True)

# --- Compute new min/max and normalize ---
min_score = df_all['RawFraudScore'].min()
max_score = df_all['RawFraudScore'].max()
score_range = max_score - min_score if max_score != min_score else 1.0

df_all['FraudScore'] = 100 * (df_all['RawFraudScore'] - min_score) / score_range

# --- Save updated CSV (with RawFraudScore and FraudScore) ---
csv_path = "fraudsummaryall.csv"
df_all.to_csv(csv_path, index=False)
print(f"Updated {csv_path} with {len(df_all)} customers and consistent normalization.")

# --- (Optional) Update MongoDB with normalized scores for new customers ---
# Get set of all CustIDs in master
master_custids = set(df_master['CustID'])

# Loop through new customers
for _, row in df_new.iterrows():
    custid = row['CustID']
    if custid not in master_custids:
        # Get full, up-to-date customer record from df_all
        record = df_all[df_all['CustID'] == custid].iloc[0].to_dict()
        fraudsummary.update_one({'CustID': custid}, {'$set': record}, upsert=True)

print("All done!")
