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

model = joblib.load('fraud_detection_model.joblib')

FEATURE_COLUMNS = [
    'TotalOrders', 'TotalReturns',
    'AOV', 'ARV', 'AccountAge', 'Rwinabuse','Rhighvalueabuse',
    'Rcycle', 'Rcategory',
    'Rvague','Rconsistency','Rdiversity',
    'FraudScore'
]

expected_features = list(model.feature_names_in_)

for doc in fraudsummary.find():
    custid = doc['CustID']

    features_df = pd.DataFrame(
        [[doc.get(col, 0) for col in expected_features]],
        columns=expected_features
    )

    pred_label = model.predict(features_df)[0]
    pred_prob = model.predict_proba(features_df)[0][1]
    
    finalfraudsummary.update_one(
        {'CustID': custid},
        {'$set': {
            'FraudLabel': bool(pred_label),
            'FraudProbability': float(pred_prob)
        }},
        upsert=True
    )
print("All customers processed. `finalfraudsummary` updated with ML predictions.")
