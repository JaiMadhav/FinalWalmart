from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import subprocess
import os

# MongoDB connection
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
    print(f"Deleted {result_orders.deleted_count} invalid orders.")
    result_fs = fraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_fs.deleted_count} invalid fraudsummary records.")
    result_finalfs = finalfraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_finalfs.deleted_count} invalid finalfraudsummary records.")

def run_scripts():
    cust_count = customers.count_documents({})
    print(f"Current customer count: {cust_count}")

    if cust_count == 0:
        print("No customers found. Skipping all scripts except cleanup.")
        return

    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    print("Pipeline completed.")

if __name__ == "__main__":
    cleanup_invalid_entries()     # Always run cleanup
    run_scripts()                 # Conditionally run the rest
