from pymongo.mongo_client import MongoClient 
from pymongo.server_api import ServerApi
from datetime import datetime
import subprocess
import pandas as pd
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
    
    result_orders = orders.delete_many({"custid": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_orders.deleted_count} invalid orders (custid not found in customers).")
    
    result_fs = fraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_fs.deleted_count} invalid fraudsummary records.")
    
    result_finalfs = finalfraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_finalfs.deleted_count} invalid finalfraudsummary records.")

def reset_files_if_no_data():
    cust_count = customers.count_documents({})
    
    if cust_count == 0:
        print("No customers or fraud data found. Cleaning up ML model and CSV file...")
        
        # Delete ML model if it exists
        if os.path.exists("fraud_detection_model.joblib"):
            os.remove("fraud_detection_model.joblib")
            print("ML model file deleted.")
        else:
            print("No ML model file to delete.")
        
        # Delete fraudsummary.csv if it exists
        if os.path.exists("fraudsummary.csv"):
            os.remove("fraudsummary.csv")
            print("fraudsummary.csv file deleted.")
        else:
            print("No fraudsummary.csv file to delete.")
        
        # Exit, nothing more to run
        exit()

def run_scripts():
    cust_count = customers.count_documents({})
    print(f"Current customer count: {cust_count}")
    
    if cust_count == 0:
        print("⚠ No customers in database. Skipping all scripts.")
        return
    
    print(f"Running pipeline at {datetime.utcnow()}...")

    # Always run DB update
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)

    if 1 <= cust_count <= 10:
        print("Customer count between 1 and 10 → Skipping ML model.")
        subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    else:
        print("Customer count > 10 → Running full pipeline with ML model.")
        subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
        subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)

    print("Pipeline execution completed.")

if __name__ == "__main__":
    cleanup_invalid_entries()
    reset_files_if_no_data()
    run_scripts()
