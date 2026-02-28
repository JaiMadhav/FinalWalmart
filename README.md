# 🛒 Walmart Fraud Detection & Customer Analysis System

A data-driven machine learning project focused on **fraud detection and customer behavior analysis** using Walmart transaction data.  
This project builds an end-to-end pipeline including data preprocessing, automation, model training, prediction generation, and CSV export workflows.

---

## 🚀 Project Overview

The goal of this project is to:

- Detect suspicious or fraudulent customer patterns
- Analyze customer transaction behavior
- Automate data processing and prediction workflows
- Generate structured outputs for analysis and reporting

The system combines **data engineering + machine learning + automation** to simulate a real-world analytics pipeline.

---

## 🧠 Key Features

- Data preprocessing and cleaning pipeline
- Automated workflow execution
- Machine learning model for pattern detection
- Customer prediction generation
- CSV export for reporting and downstream analysis
- Reusable trained models using Joblib
- Modular Python scripts for scalability

---

## 🏗️ Project Structure

```text
FinalWalmart/
│
├── PythonWalmartAutomate.py        # Automation pipeline
├── PythonWalmartDatabase.py        # Data handling & processing
├── PythonWalmartFFSCSVExport.py    # CSV export module
├── PythonWalmartMLModel.py         # ML model training & inference
│
├── fraudsummary.csv
├── fraudsummaryall.csv
├── finalfraudsummary_export.csv
├── new_customers_predictions.csv
│
├── kmeans.joblib                   # Saved ML model
├── scaler.joblib                   # Feature scaler
├── requirements.txt
└── .github/workflows/              # CI automation
```

---

## ⚙️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Automation:** Python scripts + GitHub Actions
- **Data Format:** CSV-based pipeline

---

## 📊 Machine Learning Approach

- Data preprocessing and feature scaling
- Unsupervised learning (K-Means clustering)
- Customer segmentation and anomaly pattern detection
- Prediction generation for new customer data

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/JaiMadhav/FinalWalmart.git
cd FinalWalmart
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Pipeline

```bash
python PythonWalmartAutomate.py
```

Other modules:

```bash
python PythonWalmartMLModel.py
python PythonWalmartDatabase.py
python PythonWalmartFFSCSVExport.py
```

---

## 📈 Output

The system generates:

- Fraud summary reports
- Customer prediction CSVs
- Export-ready analytics files

Example outputs:

- `fraudsummary.csv`
- `finalfraudsummary_export.csv`
- `new_customers_predictions.csv`

---

## 🎯 Learning Outcomes

- Building modular ML pipelines
- Data preprocessing and feature engineering
- Model persistence and reuse
- Automation of data workflows
- Structuring real-world analytics projects

---

## 👨‍💻 Author

**Jai Madhav**  
Computer Science Undergraduate  
AI/ML • Software Development • Data Systems  

GitHub: https://github.com/JaiMadhav

---

## ⭐ Future Improvements

- Add model evaluation metrics dashboard
- Deploy as API service
- Real-time fraud detection pipeline
- Visualization dashboard for analytics
