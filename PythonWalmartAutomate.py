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
meta = db["pipeline_meta"]

def cleanup_invalid_orders():
    valid_custids = set(customers.distinct("custid"))
    result = orders.delete_many({"custid": {"$nin": list(valid_custids)}})
    print(f"Deleted {result.deleted_count} invalid orders (custid not found in customers).")

def new_customers_exist():
    customer_ids = set(customers.distinct("custid"))
    fraud_ids = set(fraudsummary.distinct("CustID"))
    missing = customer_ids - fraud_ids
    return list(missing)

def orders_updated_since_last_check():
    record = meta.find_one({"_id": "orders_last_check"})
    last_check = record["last_checked"] if record else datetime(1970, 1, 1)
    updated = orders.count_documents({
        "$or": [
            {"createdDate": {"$gt": last_check}},
            {"updatedDate": {"$gt": last_check}}
        ]
    })
    if updated > 0:
        print(f"Detected {updated} new or updated orders since {last_check}")
        return True
    return False

def update_last_check_time():
    meta.update_one(
        {"_id": "orders_last_check"},
        {"$set": {"last_checked": datetime.utcnow()}},
        upsert=True
    )

def run_scripts():
    print(f"Triggering pipeline at {datetime.utcnow()}...")
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    print("Pipeline completed.")

if __name__ == "__main__":
    cleanup_invalid_orders()

    trigger_needed = False

    # Check for new customers
    missing_customers = new_customers_exist()
    if missing_customers:
        print(f"New customers needing processing: {missing_customers}")
        trigger_needed = True

    # Check for updated or new orders
    if orders_updated_since_last_check():
        trigger_needed = True

    if trigger_needed:
        run_scripts()
        update_last_check_time()
    else:
        print("No new customers or updated orders. No pipeline run.")
