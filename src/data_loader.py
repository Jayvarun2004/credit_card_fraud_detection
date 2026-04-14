"""
data_loader.py  -  Load and preprocess the credit card fraud dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'creditcard.csv')


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV from disk."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please download creditcard.csv from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the 'Data/' folder."
        )
    return pd.read_csv(path, engine='pyarrow')


def preprocess(df: pd.DataFrame):
    """
    Preprocess the credit card fraud dataframe.
    - Drop 'Time' column (not informative)
    - Scale 'Amount' using StandardScaler
    - Return feature matrix X, target y, and the fitted scaler
    """
    df = df.copy()
    df.drop(columns=['Time'], inplace=True)

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y, scaler


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: load -> preprocess -> stratified split.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    df = load_raw_data()
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def get_dataset_stats(path: str = DATA_PATH) -> dict:
    """
    Return high-level stats about the dataset for the dashboard.
    Note: Does NOT include raw_df to avoid doubling RAM usage.
    """
    df = load_raw_data(path)
    total = len(df)
    fraud = int(df['Class'].sum())
    legit = total - fraud
    fraud_pct = round(fraud / total * 100, 4)
    avg_amount = round(df['Amount'].mean(), 2)
    max_amount = round(df['Amount'].max(), 2)
    features = [c for c in df.columns if c != 'Class']
    # Explicitly delete the dataframe before returning to free RAM
    del df
    return {
        'total_transactions': total,
        'fraud_transactions': fraud,
        'legit_transactions': legit,
        'fraud_percentage':   fraud_pct,
        'avg_amount':         avg_amount,
        'max_amount':         max_amount,
        'features':           features,
    }
