from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import subprocess
from datetime import datetime

# MongoDB connection
uri = "mongodb+srv://jaimadhav2005:yoxZ0iSbghytySat@walmartdatabase.mwxoffr.mongodb.net/?retryWrites=true&w=majority&appName=WalmartDatabase"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["WalmartDatabase"]
customers = db["customers"]
fraudsummary = db["fraudsummary"]

def new_customers_exist():
    """
    Check if any customer exists in `customers` but not in `fraudsummary`
    """
    customer_ids = customers.distinct("custid")
    fraud_ids = fraudsummary.distinct("CustID")

    # Find customers missing in fraudsummary
    missing = set(customer_ids) - set(fraud_ids)
    return list(missing)

def run_scripts():
    """
    Run your 3 Python scripts in sequence
    """
    print(f"⚡ New customers found at {datetime.now()}, running scripts...")
    subprocess.run(["python", "C:\\Users\\BIT\\AppData\\Local\\Programs\\Python\\Python312\\PythonWalmartDatabase.py"])
    subprocess.run(["python", "C:\\Users\\BIT\\AppData\\Local\\Programs\\Python\\Python312\\PythonWalmartMLModel.py"])
    subprocess.run(["python", "C:\\Users\\BIT\\AppData\\Local\\Programs\\Python\\Python312\\PythonWalmartFinalUpdation.py"])
    print("✅ All scripts executed.")

if __name__ == "__main__":
    missing_customers = new_customers_exist()
    if missing_customers:
        print(f"📝 Missing customers: {missing_customers}")
        run_scripts()
    else:
        print("👍 No new customers needing processing.")

