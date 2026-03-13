"""
train_model.py  –  Train & compare 6 ML models on the credit card fraud dataset.
                   Saves all model artifacts to models/.

Models included:
  1. XGBoost
  2. Random Forest
  3. Logistic Regression
  4. Decision Tree
  5. K-Nearest Neighbors
  6. Naive Bayes (Gaussian)
  7. Gradient Boosting
  8. AdaBoost
"""

import os
import sys
import json
import shutil
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import get_train_test_split

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
METRICS_PATH  = os.path.join(MODELS_DIR, 'metrics.json')
FEAT_IMP_PATH = os.path.join(MODELS_DIR, 'feature_importance.json')
COMPARE_PATH  = os.path.join(MODELS_DIR, 'comparison.json')

DIVIDER = "=" * 65


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'model':            name,
        'accuracy':         round(float(accuracy_score(y_test, y_pred)), 4),
        'precision':        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1_score':         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'roc_auc':          round(float(roc_auc_score(y_test, y_prob)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'test_size':        int(len(y_test)),
        'fraud_in_test':    int(y_test.sum()),
    }


def get_feature_importance(model, feature_names):
    """Extract feature importance or coefficient magnitude."""
    if hasattr(model, 'feature_importances_'):
        fi = dict(zip(feature_names, model.feature_importances_.tolist()))
    elif hasattr(model, 'coef_'):
        fi = dict(zip(feature_names, np.abs(model.coef_[0]).tolist()))
    else:
        return None
    return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("📥 Loading and splitting data …")
    X_train, X_test, y_train, y_test, _ = get_train_test_split()
    feature_names = X_train.columns.tolist()

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2)
    print(f"   Train size : {len(y_train):,}  |  Test size : {len(y_test):,}")
    print(f"   Legit : {neg:,}  |  Fraud : {pos:,}  |  scale_pos_weight = {spw}")
    print()

    # ── Model catalogue ───────────────────────────────────────────────────────
    MODELS = {
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42,
            algorithm='SAMME',
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, class_weight='balanced', random_state=42,
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='lbfgs', random_state=42,
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1,
        ),
        'Naive Bayes': GaussianNB(),
    }

    comparison   = []
    best_auc     = -1
    best_name    = None
    best_metrics = None
    best_fi      = None

    print(DIVIDER)
    for name, model in MODELS.items():
        print(f"🚀 Training  [{name}] …")
        if name == 'XGBoost':
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=50)
        else:
            model.fit(X_train, y_train)

        m = evaluate(name, model, X_test, y_test)
        comparison.append(m)

        # Save individual model
        safe = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe}_model.pkl'))

        print(f"   ✅ AUC={m['roc_auc']}  F1={m['f1_score']}  "
              f"Precision={m['precision']}  Recall={m['recall']}")
        print()

        if m['roc_auc'] > best_auc:
            best_auc     = m['roc_auc']
            best_name    = name
            best_metrics = m
            best_fi      = get_feature_importance(model, feature_names)

    # ── Copy best model as primary ────────────────────────────────────────────
    safe_best = best_name.lower().replace(' ', '_')
    shutil.copy(
        os.path.join(MODELS_DIR, f'{safe_best}_model.pkl'),
        os.path.join(MODELS_DIR, 'xgb_model.pkl')
    )

    # ── Persist artifacts ─────────────────────────────────────────────────────
    with open(METRICS_PATH, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    with open(FEAT_IMP_PATH, 'w') as f:
        json.dump(best_fi or {}, f, indent=2)
    with open(COMPARE_PATH, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(DIVIDER)
    print(f"🏆 Best model  : {best_name}  (AUC = {best_auc})")
    print(f"📁 All models saved  → {os.path.abspath(MODELS_DIR)}")
    print(f"📊 Comparison  saved → {COMPARE_PATH}")
    print(DIVIDER)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Model':<22} {'AUC':>6} {'F1':>6} {'Recall':>8} {'Precision':>10}")
    print("-" * 55)
    for m in sorted(comparison, key=lambda x: x['roc_auc'], reverse=True):
        star = " 🏆" if m['model'] == best_name else ""
        print(f"{m['model']:<22} {m['roc_auc']:>6.4f} {m['f1_score']:>6.4f} "
              f"{m['recall']:>8.4f} {m['precision']:>10.4f}{star}")

    return comparison


if __name__ == '__main__':
    train()
