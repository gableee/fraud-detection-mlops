import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
from train import load_processed_data
import joblib
import os

# 1. Load data
X_train, X_test, y_train, y_test = load_processed_data("data/processed")

# Calculate imbalance ratio again
legit_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()
imbalance_ratio = legit_count / fraud_count

# 2. Define the base model
base_model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    random_state=42,
    eval_metric="aucpr"
)

# 3. Define the Grid (The combinations to test)
# We are testing 2 depths x 2 learning rates x 2 estimators = 8 combinations
# With 5-fold CV, that's 40 total model trainings.
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

# 4. Set up Grid Search
# scoring='average_precision' tells it to maximize AUPRC
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='average_precision', 
    cv=5, 
    verbose=2, # Prints progress
    n_jobs=-1  # Uses all CPU cores to speed it up
)

print("Starting Grid Search... (This might take a few minutes)")
grid_search.fit(X_train, y_train)

# 5. Results
print("\n--- Grid Search Complete ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation AUPRC: {grid_search.best_score_:.4f}")

# 6. Test the best model on the unseen Test Set
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_auprc = average_precision_score(y_test, y_pred_proba)
print(f"Test AUPRC with best parameters: {test_auprc:.4f}")


print("\nSaving the best tuned model...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "tuned_xgb_model.joblib")
joblib.dump(best_model, best_model_path)
print(f"Saved to {best_model_path}")