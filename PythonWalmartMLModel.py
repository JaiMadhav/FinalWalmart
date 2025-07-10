import pandas as pd
import joblib
from pymongo import MongoClient
import os

# MongoDB connection
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

# Load saved scaler and KMeans model
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')

# Fetch new customer(s) from fraudsummary (customize this query as needed)
docs = list(fraudsummary.find({}, {'_id': 0}))
df_new = pd.DataFrame(docs)

FEATURE_COLUMNS = [
    'TotalReturns', 'FraudScore', 'Rcategory', 'Rconsistency', 'Rdiversity',
    'Rvague', 'Rcycle', 'Rhighvalueabuse', 'Rwinabuse'
]

# Prepare and scale new customer data
X_new = df_new[FEATURE_COLUMNS]
X_new_scaled = scaler.transform(X_new)

# Predict clusters for new customers using saved model
clusters = kmeans.predict(X_new_scaled)
df_new['Cluster'] = clusters

def risk_level(cluster):
    if cluster == 0:
        return "high"
    elif cluster == 1:
        return "medium"
    elif cluster == 2:
        return "low"
    else:
        return "unknown"

df_new['FraudRisk'] = df_new['Cluster'].apply(risk_level)

# Upsert results into finalfraudsummary collection
for record in df_new[['CustID', 'FraudScore', 'Cluster', 'FraudRisk']].to_dict(orient='records'):
    finalfraudsummary.update_one(
        {'CustID': record['CustID']},
        {'$set': record},
        upsert=True
    )

print(f"Processed and upserted {len(df_new)} records into 'finalfraudsummary' with assigned clusters and FraudRisk.")
