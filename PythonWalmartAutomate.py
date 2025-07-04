from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import subprocess
from datetime import datetime
import os

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]
customers = db["customers"]
fraudsummary = db["fraudsummary"]

def new_customers_exist():
    customer_ids = customers.distinct("custid")
    fraud_ids = fraudsummary.distinct("CustID")

    # Find customers missing in fraudsummary
    missing = set(customer_ids) - set(fraud_ids)
    return list(missing)

def run_scripts():
    print(f"New customers found at {datetime.now()}, running scripts...")
    subprocess.run(["python", "PythonWalmartDatabase.py"], check=True)
    subprocess.run(["python", "PythonWalmartMLModel.py"], check=True)
    subprocess.run(["python", "PythonWalmartFinalUpdation.py"], check=True)
    print("All scripts executed.")

if __name__ == "__main__":
    missing_customers = new_customers_exist()
    if missing_customers:
        print(f"Missing customers: {missing_customers}")
        run_scripts()
    else:
        print("No new customers needing processing.")
