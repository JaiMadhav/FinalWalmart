# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib  # For saving the trained model

# Load preprocessed dataset from CSV file
df = pd.read_csv('fraudsummary.csv')

# Define input features used for training the model
FEATURE_COLUMNS = [
    'TotalOrders', 'TotalReturns',
    'AOV', 'ARV', 'AccountAge', 'Rwinabuse', 'Rhighvalueabuse',
    'Rcycle', 'Rcategory',
    'Rvague', 'Rconsistency', 'Rdiversity',
    'FraudScore'
]

# Define target column (label) for classification
TARGET_COLUMN = 'FraudLabel'

# Split the dataset into input features (X) and target labels (y)
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN].astype(int)  # Ensure binary classification (0 or 1)

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize Random Forest model with 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Get the probability of fraud (label=1)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print metric results
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# Print full classification report
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

# Save the trained model to a file for later use
joblib.dump(model, 'fraud_detection_model.joblib')
print("\nModel saved to fraud_detection_model.joblib")
