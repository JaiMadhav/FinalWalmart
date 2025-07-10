import pandas as pd
import joblib
from pymongo import MongoClient
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- MongoDB connection ---
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
finalfraudsummary = db["finalfraudsummary"]

# --- Load all customer data from fraudsummary.csv ---
df = pd.read_csv('fraudsummary.csv')

FEATURE_COLUMNS = [
    'TotalReturns', 'Rwinabuse', 'Rhighvalueabuse', 'Rcycle', 'Rcategory',
    'Rvague', 'Rdiversity', 'Rconsistency', 'FraudScore'
]

# --- Retrain scaler and KMeans on full dataset ---
scaler = StandardScaler()
X = df[FEATURE_COLUMNS]
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# --- Save the updated scaler and model ---
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans.joblib')

# --- Assign clusters and risk levels ---
df['Cluster'] = clusters

def risk_level(cluster):
    if cluster == 0:
        return "high"
    elif cluster == 1:
        return "medium"
    elif cluster == 2:
        return "low"
    else:
        return "unknown"

df['FraudRisk'] = df['Cluster'].apply(risk_level)

# --- Upsert results into finalfraudsummary collection ---
for record in df[['CustID', 'FraudScore', 'Cluster', 'FraudRisk']].to_dict(orient='records'):
    finalfraudsummary.update_one(
        {'CustID': record['CustID']},
        {'$set': record},
        upsert=True
    )

print(f"Retrained model and upserted {len(df)} records into 'finalfraudsummary' with clusters and FraudRisk.")
