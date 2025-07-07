from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import subprocess
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
    print(f"Deleted {result_orders.deleted_count} invalid orders.")

    result_fs = fraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_fs.deleted_count} invalid fraudsummary records.")

    result_finalfs = finalfraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_finalfs.deleted_count} invalid finalfraudsummary records.")

def reset_files_if_no_data():
    cust_count = customers.count_documents({})

    if cust_count == 0:
        print("No customers found. Resetting model and CSV...")

        # Remove content of fraudsummary.csv (empty it completely)
        try:
            with open("fraudsummary.csv", "w") as f:
                pass
            print("fraudsummary.csv emptied.")
        except Exception as e:
            print(f"Error clearing CSV: {e}")

        # Delete model file if it exists
        model_path = "fraud_detection_model.joblib"
        if os.path.exists(model_path):
            os.remove(model_path)
            print("Deleted existing ML model file.")
        else:
            print("ML model file not found. No need to delete.")

        # Exit since no customers
        exit()

def run_scripts():
    cust_count = customers.count_documents({})
    print(f"Current customer count: {cust_count}")

    if cust_count == 0:
        print("⚠ No customers found. Skipping all scripts.")
        return

    print(f"Running pipeline at {datetime.utcnow()}...")

    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)

    if 1 <= cust_count <= 10:
        print("Customer count between 1 and 10 → Skipping ML model.")
        subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    else:
        print("Customer count > 10 → Running full pipeline including ML model.")
        subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
        subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)

    print("✅ Pipeline completed.")

if __name__ == "__main__":
    cleanup_invalid_entries()
    reset_files_if_no_data()
    run_scripts()
