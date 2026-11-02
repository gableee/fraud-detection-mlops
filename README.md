# Credit Card Fraud Detection MLOps Pipeline

An end-to-end Machine Learning pipeline for detecting credit card fraud, built to simulate production environments. This project goes beyond a Jupyter Notebook by incorporating data preprocessing, hyperparameter tuning, experiment tracking (MLflow), a FastAPI backend, and Docker containerization.

## 📊 Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions (492 frauds, 0.173%)
- **Challenge:** Extreme class imbalance.

## 🏗️ Architecture & Pipeline

1. **EDA (Notebook):** Explored class imbalance, amount skewness, and time-based fraud patterns (fraud peaks at 2 AM).
2. **Preprocessing (`src/preprocess.py`):** 
   - Stratified 80/20 train/test split.
   - Log-transformed `Amount` to handle skewness.
   - Extracted `Hour` from `Time` and dropped raw time.
3. **Modeling (`src/train.py` & `src/tune.py`):** 
   - Trained an XGBoost classifier.
   - Handled imbalance using `scale_pos_weight`.
   - Tuned hyperparameters using `GridSearchCV` (optimized for AUPRC).
4. **Experiment Tracking (MLflow):** 
   - Logged parameters, metrics (AUPRC, Recall, Precision), and model artifacts.
   - Tracked both Baseline and Tuned runs side-by-side.
5. **API (`api/main.py`):**
   - Built with FastAPI.
   - Replicates preprocessing logic (`preprocess_single`) for live inference.
   - Deployed the **Baseline model** to prioritize Recall (catching fraud) over Precision, as missing a fraud costs the bank more than a false alarm.
6. **Containerization (`Dockerfile`):**
   - Packaged the API and dependencies into a `python:3.13-slim` image.

## 📂 Project Structure
```text
CC_Fraud_Detection/
├── api/
│   └── main.py               # FastAPI application and live preprocessing
├── data/
│   ├── creditcard.csv        # Raw dataset (ignored in Git)
│   └── processed/            # Train/test splits (ignored in Git)
├── models/                   # Saved .joblib model files (ignored in Git)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── preprocess.py         # Data cleaning and feature engineering
│   ├── train.py              # Baseline model training + MLflow
│   └── tune.py               # GridSearchCV hyperparameter tuning
├── Dockerfile                # Container instructions
├── requirements.txt          # Python dependencies
└── README.md
🚀 How to Run
1. Setup & Preprocess
bash
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/preprocess.py
2. Train & Track
bash
python src/train.py
python src/tune.py
# View MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db
3. Run the API (Locally)
bash
uvicorn api.main:app --reload
Go to http://127.0.0.1:8000/docs to test the endpoint.

4. Run with Docker
bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api