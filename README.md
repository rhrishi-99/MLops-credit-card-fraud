# 🚀 MLops — Credit Card Fraud Detection

A Machine Learning Operations (MLOps) project for detecting fraudulent credit card transactions using a reproducible pipeline with **MLflow** for experiment tracking and model management.

---

## 📌 Overview

This project builds an end-to-end ML pipeline to:

- Preprocess transaction data  
- Train fraud detection models  
- Track experiments with MLflow  
- Manage model versions  
- Enable reproducible training  

The dataset is highly imbalanced, making fraud detection a realistic real-world ML challenge.

---

## 🗂️ Project Structure

```
MLops-credit-card-fraud/
├── main.py                     # Main entry point
├── journal.md                  # Development notes
├── requirements.txt
├── data/                       # Dataset (not included in repo)
├── mlruns/                     # MLflow tracking artifacts
└── src/
    ├── pipeline/
    │   ├── preprocess.py       # Data preprocessing
    │   └── train.py            # Model training
    └── registry/
        └── promote.py          # Model promotion utilities
```

---

## ⚙️ Setup

### 🔹 Prerequisites

- Python 3.8+
- pip
- Kaggle account (for dataset download)

---

### 🔹 Installation

Clone the repository:

```bash
git clone https://github.com/rhrishi-99/MLops-credit-card-fraud.git
cd MLops-credit-card-fraud
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

👉 Download:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

Extract the contents into:

```
data/creditcard.csv
```

⚠️ The dataset is not included due to GitHub file size limits.

---

## ▶️ Usage

### 🔹 Run the Training Pipeline

```bash
python main.py
```

---

### 🔹 Launch MLflow UI

```bash
mlflow ui
```

Open in browser:

```
http://localhost:5000
```

You can view:

- Experiments  
- Metrics  
- Parameters  
- Artifacts  
- Model versions  

---

## 🧠 Project Components

### 🔸 Preprocessing  
`src/pipeline/preprocess.py`

- Data cleaning  
- Feature scaling  
- Handling class imbalance  
- Train/test split  

---

### 🔸 Training  
`src/pipeline/train.py`

- Model training  
- Hyperparameter tuning  
- Evaluation metrics  
- MLflow logging  

---

### 🔸 Model Registry  
`src/registry/promote.py`

- Model versioning  
- Promotion to production stages  
- Artifact management  

---

## 📁 MLflow Tracking

All experiment data is stored in:

```
mlruns/
```

This includes:

- Model artifacts  
- Metrics  
- Parameters  
- Run metadata  

---

## 📝 Development Notes

See:

```
journal.md
```

for implementation details and project progress.

---

## 🚧 Future Improvements

- Add model deployment pipeline  
- Integrate DVC for dataset versioning  
- Add CI/CD workflow  
- Real-time fraud detection API  

---

## 📜 License

Add your preferred license here (e.g., MIT License).

---

## ⭐ Acknowledgements

- ULB Machine Learning Group (dataset creators)  
- MLflow documentation  
- Kaggle community  