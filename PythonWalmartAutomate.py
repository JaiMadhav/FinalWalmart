from pymongo.mongo_client import MongoClient 
from pymongo.server_api import ServerApi
import subprocess
from datetime import datetime
import os

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]

customers = db["customers"]
orders = db["orders"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

def cleanup_invalid_entries():
    valid_custids = set(customers.distinct("custid"))
    
    # Delete orders with invalid custid
    result_orders = orders.delete_many({"custid": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_orders.deleted_count} invalid orders (custid not found in customers).")
    
    # Delete fraudsummary with invalid CustID
    result_fs = fraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_fs.deleted_count} invalid fraudsummary records.")
    
    # Delete finalfraudsummary with invalid CustID
    result_finalfs = finalfraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_finalfs.deleted_count} invalid finalfraudsummary records.")

def run_scripts():
    print(f"Running pipeline at {datetime.utcnow()}...")
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    print("All scripts executed successfully.")

if __name__ == "__main__":
    cleanup_invalid_entries()
    run_scripts()
