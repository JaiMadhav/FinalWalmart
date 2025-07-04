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
import joblib

df = pd.read_csv('fraudsummary.csv')

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] 

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'fraud_detection_model.joblib')
print("\nModel saved to fraud_detection_model.joblib")
