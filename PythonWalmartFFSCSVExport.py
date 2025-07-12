import pandas as pd
from pymongo import MongoClient
import os

# MongoDB connection
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
finalfraudsummary = db["finalfraudsummary"]

# --- Load all customer records from FFS ---
docs = list(finalfraudsummary.find({}, {'_id': 0}))
df_all = pd.DataFrame(docs)

if df_all.empty:
    print("No records found in 'finalfraudsummary'.")
    exit()
  
# --- Export to CSV ---
print("PYTHON WALMART FFS CSV EXPORT")
df_all.to_csv('finalfraudsummary_export.csv', index=False)
print("Exported finalfraudsummary collection to finalfraudsummary_export.csv")
