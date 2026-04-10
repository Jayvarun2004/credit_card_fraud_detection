"""
predict.py  -  Load saved model + scaler and run correct inference.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

_FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount']


def load_model(path: str = MODEL_PATH):
    """Load the saved model from disk."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Please run  python src/train_model.py  first."
        )
    return joblib.load(path)


def load_scaler(path: str = SCALER_PATH):
    """Load the training-fitted scaler. Falls back to None if not found (legacy)."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_transactions(df: pd.DataFrame, model=None, scaler=None, threshold: float = 0.5):
    """
    Given a DataFrame with columns V1-V28 + Amount,
    return the same DataFrame with extra columns:
      - fraud_probability  (float 0-1)
      - prediction         (0 = Legit, 1 = Fraud)
      - label              (human-readable string)

    Uses the training-fitted scaler for Amount scaling to avoid data leakage.
    """
    if model is None:
        model = load_model()

    # Load training scaler — this is correct, avoids refitting on inference data
    if scaler is None:
        scaler = load_scaler()

    data = df.copy()

    # Drop columns not used in training
    for col in ['Time', 'Class']:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Scale Amount using the fitted scaler (or fallback fit-transform if legacy)
    if scaler is not None:
        data['Amount'] = scaler.transform(data[['Amount']])
    else:
        from sklearn.preprocessing import StandardScaler
        _s = StandardScaler()
        data['Amount'] = _s.fit_transform(data[['Amount']])

    proba = model.predict_proba(data[_FEATURE_COLS])[:, 1]
    preds = (proba >= threshold).astype(int)

    result = df.copy()
    result['fraud_probability'] = proba.round(4)
    result['prediction'] = preds
    result['label'] = result['prediction'].map({0: 'Legit', 1: 'Fraud'})
    return result


def predict_single(features: dict, model=None, scaler=None, threshold: float = 0.5):
    """
    Predict fraud probability for a single transaction.
    features: dict with keys V1-V28 and Amount.
    Returns (probability, label_string).
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    df = pd.DataFrame([features])
    result = predict_transactions(df, model=model, scaler=scaler, threshold=threshold)
    prob  = float(result['fraud_probability'].iloc[0])
    label = result['label'].iloc[0]
    # Re-add emoji for display
    label = f"🚨 {label}" if prob >= threshold else f"✅ {label}"
    return prob, label
