# Import necessary libraries for database access, data processing, and system utilities
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import pandas as pd
import string
from rapidfuzz import process, fuzz
import os

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
def compute_fraud_score(row):
    safe_div = lambda n, d: n / d if d else 0  # Prevent division by zero

    # Raw weighted score (based on frequency of abuse patterns and account traits)
    raw_score = (
        1.0 * row['TotalReturns'] +
       -0.2 * row['TotalOrders'] +
       -0.001 * row['AOV'] +
        0.002 * row['ARV'] +
       -0.01 * row['AccountAge'] +
        1.0 * row['Rwinabuse'] +
        1.2 * row['Rhighvalueabuse'] +
        0.8 * row['Rcycle'] +
        0.5 * row['Rcategory'] +
        0.5 * row['Rvague'] +
        0.5 * row['Rdiversity'] +
        0.3 * row['Rconsistency']
    )

    # Proportional risk score (fraud relative to total orders or returns)
    prop_score = (
        2.0 * safe_div(row['TotalReturns'], row['TotalOrders']) +
        1.5 * safe_div(row['Rwinabuse'], row['TotalReturns']) +
        2.0 * safe_div(row['Rhighvalueabuse'], row['TotalReturns']) +
        1.0 * safe_div(row['Rcycle'], row['TotalReturns']) +
        1.0 * safe_div(row['Rcategory'], row['TotalOrders']) +
        1.0 * row['Rvague'] +
        1.0 * row['Rdiversity'] +
        1.0 * row['Rconsistency'] +
        1.0 * safe_div(row['ARV'], row['AOV']) +
        0.5 * safe_div(1, row['AccountAge'])
    )

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
    print(f"[{custid}] Raw Score: {raw_score:.2f}, Proportional Score: {prop_score:.2f}, Total Fraud Score: {fraud_score:.2f}")
    doc['FraudScore'] = float(fraud_score)
    doc['FraudLabel'] = fraud_score >= 275.0

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
