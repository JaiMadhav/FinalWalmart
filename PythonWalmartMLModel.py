import pandas as pd
import joblib
from pymongo import MongoClient
import os

# --- MongoDB connection ---
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

# --- Load data from fraudsummary collection ---
docs = list(fraudsummary.find({}, {'_id': 0}))
df_all = pd.DataFrame(docs)

if df_all.empty:
    print("No customers found in 'fraudsummary'.")
    exit()

# --- Feature columns used for prediction ---
FEATURE_COLUMNS = [
    'TotalReturns', 'Rwinabuse', 'Rhighvalueabuse', 'Rcycle', 'Rcategory',
    'Rvague', 'Rdiversity', 'Rconsistency', 'FraudScore'
]

# --- Load pretrained scaler and KMeans model ---
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')

# --- Predict clusters and assign fraud risk ---
X = df_all[FEATURE_COLUMNS]
X_scaled = scaler.transform(X)
clusters = kmeans.predict(X_scaled)
df_all['Cluster'] = clusters

def risk_level(cluster):
    return {0: "high", 1: "low", 2: "medium"}.get(cluster, "unknown")

df_all['FraudRisk'] = df_all['Cluster'].apply(risk_level)

# --- Upsert results into finalfraudsummary ---
COLUMNS_TO_SAVE = ['CustID', 'FraudScore', 'Cluster', 'FraudRisk']
for record in df_all[COLUMNS_TO_SAVE].to_dict(orient='records'):
    finalfraudsummary.update_one(
        {'CustID': record['CustID']},
        {'$set': record},
        upsert=True
    )

print(f"Processed and upserted {len(df_all)} customers from 'fraudsummary' into 'finalfraudsummary'.")
