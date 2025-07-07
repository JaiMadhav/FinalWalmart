from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import joblib
import os

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]
customers = db["customers"]

# Count customers
cust_count = customers.count_documents({})
use_ml = cust_count > 10

if use_ml:
    model = joblib.load('fraud_detection_model.joblib')
    expected_features = list(model.feature_names_in_)
    print(f"{cust_count} customers found — using ML model.")
else:
    print(f"{cust_count} customers found — using fraud score logic (no ML).")

# Process each fraudsummary doc
for doc in fraudsummary.find():
    custid = doc['CustID']
    
    if use_ml:
        # Prepare features
        features_df = pd.DataFrame(
            [[doc.get(col, 0) for col in expected_features]],
            columns=expected_features
        )
        pred_label = model.predict(features_df)[0]
        pred_prob = model.predict_proba(features_df)[0][1]
    else:
        pred_label = doc.get('FraudLabel', False)
        pred_prob = 0.0

    finalfraudsummary.update_one(
        {'CustID': custid},
        {'$set': {
            'FraudLabel': bool(pred_label),
            'FraudProbability': float(pred_prob)
        }},
        upsert=True
    )

print("Final fraud summary updated.")
