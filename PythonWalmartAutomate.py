# Import required libraries
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import subprocess
import os

# Load MongoDB URI from environment variable for secure access
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]

# Connect to relevant MongoDB collections
customers = db["customers"]
orders = db["orders"]
fraudsummary = db["fraudsummary"]
finalfraudsummary = db["finalfraudsummary"]

# Remove database entries that reference invalid/non-existent customers
def cleanup_invalid_entries():
    valid_custids = set(customers.distinct("custid"))

    # Delete orders with unknown customer IDs
    result_orders = orders.delete_many({"custid": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_orders.deleted_count} invalid orders.")

    # Delete fraudsummary records with unknown customer IDs
    result_fs = fraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_fs.deleted_count} invalid fraudsummary records.")

    # Delete finalfraudsummary records with unknown customer IDs
    result_finalfs = finalfraudsummary.delete_many({"CustID": {"$nin": list(valid_custids)}})
    print(f"Deleted {result_finalfs.deleted_count} invalid finalfraudsummary records.")

# Reset files and model artifacts if no customer data exists
def reset_files_if_no_data():
    cust_count = customers.count_documents({})

    if cust_count == 0:
        print("No customers found. Resetting model and CSV...")

        # Empty the fraudsummary.csv file (removes all data)
        try:
            with open("fraudsummary.csv", "w") as f:
                pass
            print("fraudsummary.csv emptied.")
        except Exception as e:
            print(f"Error clearing CSV: {e}")

# Main function to run pipeline scripts based on customer volume
def run_scripts():
    cust_count = customers.count_documents({})
    print(f"Current customer count: {cust_count}")

    # Abort execution if there is no customer data
    if cust_count == 0:
        print("No customers found. Skipping all scripts.")
        return

    # Step 1: Import data from raw sources and update MongoDB
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)

    # Step 2: Always run all scripts if there is at least 1 customer
    print("Running full pipeline including ML model (regardless of customer count).")
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)

    print("Pipeline completed.")

# Entry point: Cleanup, reset checks, and run full pipeline
if __name__ == "__main__":
    cleanup_invalid_entries()     # Remove orphaned records
    reset_files_if_no_data()      # Handle empty database scenario
    run_scripts()                 # Execute processing pipeline
