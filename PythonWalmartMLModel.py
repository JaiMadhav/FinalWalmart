import pandas as pd
import joblib
from pymongo import MongoClient
import os

print("KMEANS CLUSTER ASSIGNMENT USING SAVED MODEL (MONGODB):")

# MongoDB connection
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]

# Fetch all documents from the fraudsummary collection, include all columns except MongoDB's _id
docs = list(fraudsummary.find({}, {'_id': 0}))

# Convert to DataFrame
df = pd.DataFrame(docs)

# --- If you want to include previously excluded columns in the final output, do nothing extra here.
# They are already present in df if they exist in MongoDB.

# Define features used in your trained model (update as needed)
FEATURE_COLUMNS = [
    'TotalReturns', 'FraudScore', 'Rcategory', 'Rconsistency', 'Rdiversity',
    'Rvague', 'Rcycle', 'Rhighvalueabuse', 'Rwinabuse'
    # Make sure this matches your actual model's features and order
]

# Load the saved scaler and KMeans model
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')

# Prepare and scale features
X = df[FEATURE_COLUMNS]
X_scaled = scaler.transform(X)

# Assign clusters using the loaded model
clusters = kmeans.predict(X_scaled)
df['Cluster'] = clusters

# --- Save only the columns you want (including previously excluded ones) ---
# For example, to include 'CustID', 'FraudScore', and 'Cluster':
df[['CustID', 'FraudScore', 'Cluster']].to_csv('fraudsummary_custid_fraudscore_cluster.csv', index=False)

# Or, to include ALL columns (including those previously excluded):
df.to_csv('fraudsummary_with_clusters.csv', index=False)

print("Cluster assignments saved to fraudsummary_with_clusters.csv")
