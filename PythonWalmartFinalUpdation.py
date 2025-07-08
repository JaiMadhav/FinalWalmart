# Import required libraries
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import joblib
import os

# Load MongoDB URI from environment variable
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))

# Connect to database and required collections
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]
customers = db["customers"]

# Count number of customers to decide if ML model should be used
cust_count = customers.count_documents({})
use_ml = cust_count > 10  # Use ML only if we have enough data

# Load model if using ML, else use rule-based fraud labels
if use_ml:
    model = joblib.load('fraud_detection_model.joblib')  # Load trained model
    expected_features = list(model.feature_names_in_)    # Get expected feature columns
    print(f"{cust_count} customers found — using ML model.")
else:
    print(f"{cust_count} customers found — using fraud score logic (no ML).")

# Process each document in fraudsummary collection
for doc in fraudsummary.find():
    custid = doc['CustID']  # Unique customer ID
    
    if use_ml:
        # Prepare feature DataFrame from the document (default 0 if key is missing)
        features_df = pd.DataFrame(
            [[doc.get(col, 0) for col in expected_features]],
            columns=expected_features
        )
        
        # Predict fraud label and probability using the trained model
        pred_label = model.predict(features_df)[0]
        pred_prob = model.predict_proba(features_df)[0][1]
    else:
        # If not using ML, fall back to fraud label from existing logic
        pred_label = doc.get('FraudLabel', False)
        pred_prob = 0.0  # Set probability to zero for non-ML case

    # Update or insert the final fraud prediction into finalfraudsummary collection
    finalfraudsummary.update_one(
        {'CustID': custid},
        {'$set': {
            'FraudLabel': bool(pred_label),
            'FraudProbability': float(pred_prob)
        }},
        upsert=True
    )

# Final status message
print("Final fraud summary updated.")
