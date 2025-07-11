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
customers = db["customers"]  # Collection with new customers
finalfraudsummary = db["finalfraudsummary"]

# --- Load all customer data from fraudsummaryall.csv ---
df_all = pd.read_csv('fraudsummaryall.csv')
df_master = pd.read_csv('fraudsummary.csv')

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

# --- Identify new customers (not in master) ---
master_custids = set(df_master['CustID'])
df_new = df_all[~df_all['CustID'].isin(master_custids)].copy()  # <-- .copy() prevents SettingWithCopyWarning

if df_new.empty:
    print("No new customers to process.")
else:
    # --- Predict clusters for new customers ---
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

    # --- Save predictions for new customers to CSV (optional) ---
    df_new.to_csv('new_customers_predictions.csv', index=False)
    
    COLUMNS_TO_SAVE = ['CustID', 'FraudScore', 'Cluster', 'FraudRisk']
    for record in df_new[COLUMNS_TO_SAVE].to_dict(orient='records'):
        finalfraudsummary.update_one(
            {'CustID': record['CustID']},
            {'$set': record},
            upsert=True
        )

    print(f"Processed and upserted {len(df_new)} NEW customer records into 'finalfraudsummary' with clusters and FraudRisk.")
