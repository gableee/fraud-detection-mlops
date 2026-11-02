import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from train import load_processed_data

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"data/{filename}", dpi=100, bbox_inches="tight")
    plt.show()

def plot_pr_curve(y_true, y_proba, model_name, color):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    plt.plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.3f})', color=color, linewidth=2)

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data("data/processed")
    
    # Load models
    baseline_model = joblib.load("models/baseline_xgb_model.joblib")
    tuned_model = joblib.load("models/tuned_xgb_model.joblib")
    
    # Get predictions and probabilities
    y_pred_base = baseline_model.predict(X_test)
    y_proba_base = baseline_model.predict_proba(X_test)[:, 1]
    
    y_pred_tuned = tuned_model.predict(X_test)
    y_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]
    
    # 1. Plot Confusion Matrices
    plot_confusion_matrix(y_test, y_pred_base, "Baseline Model Confusion Matrix", "cm_baseline.png")
    plot_confusion_matrix(y_test, y_pred_tuned, "Tuned Model Confusion Matrix", "cm_tuned.png")
    
    # 2. Plot Precision-Recall Curves (Comparing both)
    plt.figure(figsize=(8, 6))
    plot_pr_curve(y_test, y_proba_base, "Baseline", "gray")
    plot_pr_curve(y_test, y_proba_tuned, "Tuned", "blue")
    plt.xlabel('Recall (Percentage of Frauds Caught)')
    plt.ylabel('Precision (Trustworthiness of Alarms)')
    plt.title('Precision-Recall Curve: Baseline vs Tuned')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("data/pr_curve_comparison.png", dpi=100, bbox_inches="tight")
    plt.show()