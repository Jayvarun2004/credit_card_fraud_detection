# 🛡️ FraudGuard: Credit Card Fraud Detection

An enterprise-grade, Splunk-style AI dashboard for detecting and analysing fraudulent credit card transactions. Built with Python, Streamlit, XGBoost, and an autonomous AI Copilot powered by Groq LLaMA-3.

![Splunk Dashboard Overview](https://img.shields.io/badge/UI-Splunk%20Style-0877a6?style=for-the-badge&logo=splunk)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EC6C2D?style=for-the-badge)
![Groq](https://img.shields.io/badge/AI_Copilot-Groq_LLaMA3-8B00FF?style=for-the-badge)

---

## ✨ Features

### 📊 Dashboard Pages
| Page | Description |
|------|----|
| 🏠 **Overview** | KPI cards, fraud distribution, transaction histograms, 3D Geo Globe, Fraud Ring Network Graph |
| 📊 **Model Performance** | ROC Curves, Confusion Matrix, Radar Chart, grouped bar, feature importance for all 6 models |
| 🔍 **Manual Predict** | Enter V1–V28 features, adjust live decision threshold slider, get SHAP Waterfall explanation |
| 📁 **Batch Analysis** | Upload CSV → run bulk predictions → view pie chart + top offenders → download results |
| 📡 **Live Monitor** | Real-time simulated transaction feed with animated threat alerts |
| ⚙️ **MLOps & Retraining** | One-click model retraining trigger from the UI — no terminal needed |
| 🤖 **AI Copilot** | Groq LLaMA-3.3-70b powered fraud analyst chatbot with live dataset context |

### 🧠 ML & AI
- Trains **6 models**: XGBoost, Random Forest, Gradient Boosting, AdaBoost, Decision Tree, Logistic Regression
- Automatically selects the **best model** by ROC-AUC and saves it as the primary
- **SHAP Explainable AI** — Waterfall chart for every manual prediction
- **Correctly saved StandardScaler** — no data leakage in inference pipeline
- One-click **Executive PDF Report** generation (KPIs + model metrics)

### 🌍 Visualisations
- Rotating **3D Fraud Origin Globe** using Plotly Scattergeo
- **Fraud Ring Network Graph** using NetworkX — shows transaction similarity clusters
- ROC Curves for all models rendered simultaneously

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the Dataset
- Download `creditcard.csv` from [Kaggle ULB Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Place it inside the `Data/` folder:
```
Data/
└── creditcard.csv
```

### 4. Configure AI Copilot (optional)
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Train the Models
```bash
python src/train_model.py
```
> Takes ~5–8 minutes. Generates `.pkl` and `.json` files in `models/`.

### 6. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```text
credit_card_fraud_detection/
├── app.py                      # Main Streamlit application (7 pages)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container config
├── docker-compose.yml          # Docker compose config
├── .env.example                # API key template
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── train_model.py          # ML training pipeline (6 models)
│   ├── predict.py              # Inference with correct scaler loading
│   ├── xai.py                  # SHAP Explainable AI waterfall charts
│   ├── geo_mock.py             # Deterministic Lat/Lon geo-mapping
│   ├── network_graph.py        # NetworkX fraud ring graph builder
│   ├── report_gen.py           # fpdf2 Executive PDF report generator
│   └── components/
│       └── live_monitor.html   # Animated live transaction monitor
├── models/                     # Generated after training (gitignored)
│   ├── xgb_model.pkl           # Best model (primary)
│   ├── scaler.pkl              # Training-fitted StandardScaler
│   ├── metrics.json            # Best model metrics
│   ├── comparison.json         # All 6 models comparison data
│   └── feature_importance.json # Top feature importances
└── Data/                       # Dataset directory (gitignored)
    └── creditcard.csv
```

---

## 🐳 Docker
```bash
docker-compose up --build
# Dashboard available at http://localhost:8511
```

---

## 📜 License
This project is for educational and portfolio purposes.
