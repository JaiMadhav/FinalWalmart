import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib

# Load data
df = pd.read_csv('fraudsummary.csv')

# Features & target
FEATURE_COLUMNS = [
    'TotalOrders', 'TotalReturns',
    'AOV', 'ARV', 'AccountAge', 'Rwinabuse','Rhighvalueabuse',
    'Rcycle', 'Rcategory',
    'Rvague','Rconsistency','Rdiversity',
    'FraudScore'
]
TARGET_COLUMN = 'FraudLabel'

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n✅ Confusion Matrix (Actual rows x Predicted columns):")
print(cm)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n✅ Accuracy       : {accuracy:.4f}")
print(f"✅ Precision      : {precision:.4f}")
print(f"✅ Recall         : {recall:.4f}")
print(f"✅ F1-score       : {f1:.4f}")
print(f"✅ ROC AUC        : {roc_auc:.4f}")

# Detailed classification report
print("\n✅ Detailed classification report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'fraud_detection_model.joblib')
print("\n✅ Model saved to fraud_detection_model.joblib")
