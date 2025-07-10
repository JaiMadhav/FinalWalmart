import pandas as pd
import joblib
from pymongo import MongoClient
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("KMEANS MODEL RETRAINING AND ASSIGNMENT (MONGODB):")

# MongoDB connection
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

# Fetch all documents from the fraudsummary collection
docs = list(fraudsummary.find({}, {'_id': 0}))

# Convert to DataFrame
df = pd.DataFrame(docs)

# Define features used in your trained model
FEATURE_COLUMNS = [
    'TotalReturns', 'FraudScore', 'Rcategory', 'Rconsistency', 'Rdiversity',
    'Rvague', 'Rcycle', 'Rhighvalueabuse', 'Rwinabuse'
]

# Prepare features
X = df[FEATURE_COLUMNS]

# Retrain scaler and KMeans model on all current data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# You can change n_clusters as needed
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Save the updated scaler and model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans.joblib')
print("Scaler and KMeans model retrained and saved.")

# Assign clusters to DataFrame
df['Cluster'] = clusters

# Add FraudRisk attribute
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

# Upsert into finalfraudsummary collection
for record in df[['CustID', 'FraudScore', 'Cluster', 'FraudRisk']].to_dict(orient='records'):
    finalfraudsummary.update_one(
        {'CustID': record['CustID']},
        {'$set': record},
        upsert=True
    )

print(f"Upserted {len(df)} records into 'finalfraudsummary' with FraudRisk.")
