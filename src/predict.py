"""
predict.py  –  Load saved XGBoost model and run inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model.pkl')

_FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount']


def load_model(path: str = MODEL_PATH):
    """Load the saved XGBoost model from disk."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Please run  python src/train_model.py  first."
        )
    return joblib.load(path)


def predict_transactions(df: pd.DataFrame, model=None, threshold: float = 0.5):
    """
    Given a DataFrame with columns V1–V28 + Amount,
    return the same DataFrame with two extra columns:
      - fraud_probability  (float 0–1)
      - prediction         (0 = Legit, 1 = Fraud)
    """
    if model is None:
        model = load_model()

    data = df.copy()

    # Scale Amount if not already scaled
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])

    # Keep only expected features (drop Time if present)
    for col in ['Time', 'Class']:
        if col in data.columns:
            data = data.drop(columns=[col])

    proba = model.predict_proba(data[_FEATURE_COLS])[:, 1]
    preds = (proba >= threshold).astype(int)

    result = df.copy()
    result['fraud_probability'] = proba.round(4)
    result['prediction'] = preds
    result['label'] = result['prediction'].map({0: '✅ Legit', 1: '🚨 Fraud'})
    return result


def predict_single(features: dict, model=None, threshold: float = 0.5):
    """
    Predict fraud probability for a single transaction.
    features: dict with keys V1–V28 and Amount.
    Returns (probability, label).
    """
    if model is None:
        model = load_model()

    df = pd.DataFrame([features])
    result = predict_transactions(df, model=model, threshold=threshold)
    prob = float(result['fraud_probability'].iloc[0])
    label = result['label'].iloc[0]
    return prob, label
