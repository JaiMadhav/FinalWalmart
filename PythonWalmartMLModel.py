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
customers = db["customer"]  # Collection with new customers

# --- Load all customer data from fraudsummaryall.csv ---
df_all = pd.read_csv('fraudsummaryall.csv')

FEATURE_COLUMNS = [
    'TotalReturns', 'Rwinabuse', 'Rhighvalueabuse', 'Rcycle', 'Rcategory',
    'Rvague', 'Rdiversity', 'Rconsistency', 'FraudScore'
]

# --- Retrain scaler and KMeans on full dataset ---
scaler = StandardScaler()
X = df_all[FEATURE_COLUMNS]
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
kmeans.fit(X_scaled)

# --- Save the updated scaler and model ---
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans.joblib')

# --- Load new customers from MongoDB ---
docs = list(customers.find({}, {'_id': 0}))
df_new = pd.DataFrame(docs)
if df_new.empty:
    print("No new customers found. Exiting.")
    exit()

# --- Only process customers not already in master ---
master_custids = set(df_all['CustID'])
df_new = df_new[~df_new['CustID'].isin(master_custids)]

if df_new.empty:
    print("All new customers already exist in master. Nothing to process.")
    exit()

# --- Scale and predict cluster for new customers ---
X_new = df_new[FEATURE_COLUMNS]
X_new_scaled = scaler.transform(X_new)
clusters_new = kmeans.predict(X_new_scaled)
df_new['Cluster'] = clusters_new

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

# --- Upsert only new customers into finalfraudsummary collection ---
for record in df_new[['CustID', 'FraudScore', 'Cluster', 'FraudRisk']].to_dict(orient='records'):
    finalfraudsummary.update_one(
        {'CustID': record['CustID']},
        {'$set': record},
        upsert=True
    )

print(f"Processed and upserted {len(df_new)} NEW customer records into 'finalfraudsummary' with clusters and FraudRisk.")
