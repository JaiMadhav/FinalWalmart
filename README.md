# FinalWalmart

FinalWalmart is an automated fraud detection that processes customer and order data, computes fraud scores, trains a machine learning model, and updates predictions. The system is designed to work with MongoDB Atlas and is integrated with GitHub Actions for scheduled automation.

## Overview

The repository contains Python scripts that:

* Connect to MongoDB Atlas to fetch and update customer and order data.
* Compute fraud scores using both rule-based metrics and proportional calculations.
* Train a Random Forest model to predict fraud labels.
* Apply the trained model to update fraud predictions in the database.
* Export the fraud summary as a CSV file and save the trained model.

The pipeline is triggered daily at a specified time using GitHub Actions, with the option for manual execution.

## Repository Structure

```
FinalWalmart/
├── PythonWalmartAutomate.py          # Master controller script that runs the entire pipeline
├── PythonWalmartDatabase.py          # Computes fraud scores and updates MongoDB and CSV
├── PythonWalmartMLModel.py           # Trains the machine learning model and saves it
├── PythonWalmartFinalUpdation.py     # Applies the model and updates predictions in MongoDB
├── fraudsummary.csv                  # Auto-generated fraud summary data
├── fraud_detection_model.joblib      # Auto-generated machine learning model
├── requirements.txt                  # List of Python dependencies
└── .github/workflows/main.yml        # GitHub Actions workflow configuration
```

## How the Pipeline Works

1. The master script checks for new or updated customers in MongoDB.
2. If new data is detected, it updates fraud metrics and exports a refreshed CSV file.
3. The machine learning model is retrained using the updated data.
4. Predictions are applied and the database is updated.
5. The updated CSV file and model are uploaded as artifacts or optionally committed to the repository.
6. The fraud detection pipeline is designed to work as follows:

Input Collections

`customers`: Contains customer profile data. Example fields: `custid`, `name`, `phone_no`, `email`, `createdDate`.

`orders`: Contains individual order data linked to a customer. Example fields: `order_id`, `custid`, `transaction_value`, `order_date`, `return_label`, `return_date`, `return_category`, `return_item_id`, `return_reason`.

Processing

The pipeline reads from `customers` and `orders` collections.

It generates fraud-related metrics (e.g., return rates, patterns) and stores them in the `fraudsummary` collection.

The machine learning model is trained on the data in `fraudsummary` and generates predictions.

Output

The final fraud predictions for each customer are saved in the `finalfraudsummary` collection.

This collection contains fields such as `CustID`, `FraudLabel` (predicted fraud/not fraud), and `FraudProbability`.

## Developer Instructions
Feed data into `customers` and `orders`

* Ensure that new or updated customer and order data is correctly inserted into these collections.

Fetch data from `finalfraudsummary`

* The backend should query only `finalfraudsummary` to get fraud detection results.

## Automation

* Manual execution is available through GitHub Actions.

## Setup Requirements

* The MongoDB connection URI must be stored as a GitHub secret named `MONGO_URI`.
* MongoDB collections: `customers`, `orders`, `fraudsummary`, and `finalfraudsummary` should be set up.
* The pipeline requires Python 3.10 and dependencies listed in `requirements.txt`.

## Notes

* The CSV file and model are saved as artifacts in each workflow run.
* Direct commits to the repository from the workflow require appropriate permissions.
