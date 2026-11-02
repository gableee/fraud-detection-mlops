import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Loads the CSV dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows.")
    return df

def engineer_features(df):
    """Creates the 'Hour' feature and drops the raw 'Time' column."""
    df["Hour"] = (df["Time"] / 3600 % 24).astype(int)
    df = df.drop(columns=["Time"])
    return df

def scale_features(df):
    """Applies log1p transformation to the Amount column."""
    df["Amount"] = np.log1p(df["Amount"])
    return df

def split_and_save(df, output_dir):
    """
    Splits the data (stratified 80/20) and saves the results to CSV.
    """
    print("Splitting data (80/20 stratified)...")
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # The split we discussed: shuffle=True to mix time, stratify=y to keep 0.173% fraud
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        shuffle=True, 
        stratify=y, 
        random_state=42
    )
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    print("Saving processed data...")
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Print a summary
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    
    # Verify the stratification worked
    train_fraud_rate = y_train.mean() * 100
    test_fraud_rate = y_test.mean() * 100
    print(f"\nFraud rate in Train: {train_fraud_rate:.3f}%")
    print(f"Fraud rate in Test:  {test_fraud_rate:.3f}%")

if __name__ == "__main__":
    raw_data_path = "data/creditcard.csv"
    processed_dir = "data/processed"

    
    df = load_data(raw_data_path)
    df = engineer_features(df)
    df = scale_features(df)
    split_and_save(df, processed_dir)
    
    print("\nPreprocessing complete! Ready for training.")