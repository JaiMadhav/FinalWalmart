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

def cleanup_invalid_orders():
    valid_custids = set(customers.distinct("custid"))
    result = orders.delete_many({"custid": {"$nin": list(valid_custids)}})
    print(f"Deleted {result.deleted_count} invalid orders (custid not found in customers).")

def run_scripts():
    print(f"Running pipeline at {datetime.utcnow()}...")
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    print("All scripts executed successfully.")

if __name__ == "__main__":
    cleanup_invalid_orders()
    run_scripts()
