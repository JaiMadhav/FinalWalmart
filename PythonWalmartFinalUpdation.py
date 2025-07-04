from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import joblib

# MongoDB connection
uri = "mongodb+srv://jaimadhav2005:yoxZ0iSbghytySat@walmartdatabase.mwxoffr.mongodb.net/?retryWrites=true&w=majority&appName=WalmartDatabase"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

# Load saved ML model
model = joblib.load('fraud_detection_model.joblib')

# Feature columns as expected by the model
FEATURE_COLUMNS = [
    'TotalOrders', 'TotalReturns',
    'AOV', 'ARV', 'AccountAge', 'Rwinabuse','Rhighvalueabuse',
    'Rcycle', 'Rcategory',
    'Rvague','Rconsistency','Rdiversity',
    'FraudScore'
]

# Process each document
# Get model's expected feature names
expected_features = list(model.feature_names_in_)

for doc in fraudsummary.find():
    custid = doc['CustID']

    # Build DataFrame matching model's expected columns
    features_df = pd.DataFrame(
        [[doc.get(col, 0) for col in expected_features]],
        columns=expected_features
    )

    # Apply ML model
    pred_label = model.predict(features_df)[0]
    pred_prob = model.predict_proba(features_df)[0][1]

    # Update finalfraudsummary
    finalfraudsummary.update_one(
        {'CustID': custid},
        {'$set': {
            'FraudLabel': bool(pred_label),
            'FraudProbability': float(pred_prob)
        }},
        upsert=True
    )


print("✅ All customers processed. `finalfraudsummary` updated with ML predictions.")
