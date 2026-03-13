# 🛡️ FraudGuard: Credit Card Fraud Detection

An enterprise-grade, Splunk-style dashboard for detecting and analyzing fraudulent credit card transactions. Built with Python, Streamlit, and Scikit-Learn utilizing multiple machine learning models.

![Splunk Dashboard Overview](https://img.shields.io/badge/UI-Splunk%20Style-0877a6?style=for-the-badge&logo=splunk)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)

## ✨ Features
* **Splunk-Inspired UI:** Dark theme, glowing KPIs, interactive Plotly charts, and severity badges (`CRITICAL`, `HIGH`, `LOW`).
* **Multi-Model Support:** Trains and compares 8 machine learning models including XGBoost, Random Forest, AdaBoost, and more.
* **Interactive Prediction:** Allows manual entry of transaction details and batch CSV uploads for instant fraud analysis.
* **Extensive Metrics:** Detailed view of Accuracy, Precision, Recall, F1-Score, and ROC-AUC for all models.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\\Scripts\\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   - Download the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Create a `Data/` folder in the root directory and place `creditcard.csv` inside it.

4. **Train the Models:**
   ```bash
   python src/train_model.py
   ```
   *(This step takes a few minutes and will generate `.pkl` and `.json` files in the `models/` directory).*

5. **Start the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
```text
credit_card_fraud_detection/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/                    
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── train_model.py      # ML Model training pipeline
│   └── predict.py          # Prediction inference functions
├── models/                 # Generated models and metrics (Created after training)
└── Data/                   # Ignored by git (Place your creditcard.csv here)
```

## 📜 License
This project is for educational and portfolio purposes.
