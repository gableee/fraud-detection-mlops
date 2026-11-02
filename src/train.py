from fileinput import filename

import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import os
import joblib
import mlflow
import mlflow.xgboost
    

def load_processed_data(data_dir):
    """Loads the train/test splits we created in Phase 2."""
    print("Loading processed data...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze() # squeeze turns DataFrame to Series
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training XGBoost model...")
    
    # The imbalance ratio: 578 legit for every 1 fraud
    # We calculate it dynamically just in case the data changes
    legit_count = (y_train == 0).sum()
    fraud_count = (y_train == 1).sum()
    imbalance_ratio = legit_count / fraud_count
    
    print(f"Imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")
    
    # Initialize XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,            # Number of trees to build
        max_depth=4,                 # How deep each tree can go (prevents overfitting)
        learning_rate=0.1,           # How much each tree corrects the previous one
        scale_pos_weight=imbalance_ratio, # OUR IMBALANCE FIX
        random_state=42,
        eval_metric="aucpr"          # We tell XGBoost to optimize for AUPRC internally
    )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on Test Set...")
    
    # Get predictions (0 or 1)
    y_pred = model.predict(X_test)
    
    # Get probabilities (0.0 to 1.0) for AUPRC calculation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit (0)", "Fraud (1)"]))
    
    # Print Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"\nAUPRC Score: {auprc:.4f}")


def save_model(model, output_dir, filename="baseline_xgb_model.joblib"):
    """Saves the trained model to disk."""
    print(f"\nSaving model to {output_dir}/{filename}...")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)
    print("Model saved successfully!")


if __name__ == "__main__":
    
    # 1.  Connect to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Fraud_Detection")
    
    X_train, X_test, y_train, y_test = load_processed_data("data/processed")

    # Calculate imbalance ratio
    legit_count = (y_train == 0).sum()
    fraud_count = (y_train == 1).sum()
    imbalance_ratio = legit_count / fraud_count
    
    base_model = xgb.XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric="aucpr"
    )
    
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision', 
        cv=5, 
        verbose=2, 
        n_jobs=-1
    )
    
    print("Starting Grid Search...")

    # 2. Start the MLflow Run
    with mlflow.start_run(run_name="Tuned_XGBoost_GridSearch"):
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # 3. Evaluate on Test Set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_auprc = average_precision_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nBest Parameters: {best_params}")
        print(f"Test AUPRC: {test_auprc:.4f}")

        # 4. Log parameters and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("auprc", test_auprc)
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("precision_fraud", report["1"]["precision"])

        # 5. Save and Log the Model
        model_output_dir = "models"
        os.makedirs(model_output_dir, exist_ok=True)
        joblib.dump(best_model, f"{model_output_dir}/tuned_xgb_model.joblib")
        mlflow.xgboost.log_model(best_model, "tuned_xgb_model")

        print("MLflow logging complete. Model saved and logged successfully!")