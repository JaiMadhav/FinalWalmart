from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import pandas as pd
import string
from rapidfuzz import process, fuzz

# MongoDB connection URI
uri = "mongodb+srv://jaimadhav2005:yoxZ0iSbghytySat@walmartdatabase.mwxoffr.mongodb.net/?retryWrites=true&w=majority&appName=WalmartDatabase"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

db = client["WalmartDatabase"]
customers = db["customers"]
orders = db["orders"]
fraudsummary = db["fraudsummary"]

# Base categories for return reasons
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

def normalize_reason(reason):
    reason = reason.lower()
    reason = reason.translate(str.maketrans('', '', string.punctuation))
    return reason.strip()

def map_to_base(norm_reason):
    match, score, _ = process.extractOne(norm_reason, base_categories.keys(), scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else "other"

def days_between(date1, date2):
    return (date2 - date1).days

def compute_fraud_score(row):
    safe_div = lambda n, d: n / d if d else 0
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
    return raw_score + prop_score

print(f"🔄 Running fraud summary update at {datetime.now()}")
today = datetime.now()

for cust in customers.find():
    custid = cust['custid']
    account_age = days_between(cust['createdDate'], today)
    cust_orders = list(orders.find({'custid': custid}))

    total_orders = len(cust_orders)
    total_returns = sum(1 for o in cust_orders if o.get('return_label'))
    aov = sum(o['transaction_value'] for o in cust_orders) / total_orders if total_orders else 0
    ret_orders = [o for o in cust_orders if o.get('return_label')]
    arv = sum(o['transaction_value'] for o in ret_orders) / len(ret_orders) if ret_orders else 0

    rwinabuse = sum(1 for o in ret_orders if days_between(o['order_date'], o['return_date']) > 75)
    rhighvalueabuse = sum(1 for o in ret_orders if o['transaction_value'] > 1.5 * aov)

    item_counts = {}
    for o in ret_orders:
        iid = o.get('return_item_id')
        if iid:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    rcycle = sum(1 for v in item_counts.values() if v > 1)

    categories = set(o.get('return_category') for o in ret_orders if o.get('return_category'))
    rcategory = len(categories)

    reason_data = []
    for o in ret_orders:
        reason = o.get('return_reason', '')
        norm = normalize_reason(reason)
        base = map_to_base(norm)
        vague = base_categories[base]
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

    fraud_score = compute_fraud_score(doc)
    fraud_score = max(fraud_score, 0)
    doc['FraudScore'] = float(fraud_score)
    doc['FraudLabel'] = fraud_score >= 12.0

    fraudsummary.update_one({'CustID': custid}, {'$set': doc}, upsert=True)

    # Export to CSV
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
    print("✅ Exported fraudsummary.csv successfully.")

print("✅ Fraud summary updated in MongoDB.")
