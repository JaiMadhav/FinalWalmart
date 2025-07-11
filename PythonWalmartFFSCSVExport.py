import pandas as pd
from pymongo import MongoClient
import os

# MongoDB connection
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["WalmartDatabase"]
finalfraudsummary = db["finalfraudsummary"]

# Fetch all documents (excluding MongoDB's _id field)
docs = list(finalfraudsummary.find({}, {'_id': 0}))
print("PYTHON WALMART FFS CSV EXPORT")
# Convert to DataFrame
df = pd.DataFrame(docs)
print("Exporting finalfraudsummary collection to csv")
# Save to CSV (all columns)
df.to_csv('finalfraudsummary_export.csv', index=False)
print("Exported finalfraudsummary collection to finalfraudsummary_export.csv")
