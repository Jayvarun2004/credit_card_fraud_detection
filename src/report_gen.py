"""
report_gen.py - PDF Report Generation using fpdf2
"""

from fpdf import FPDF
from datetime import datetime

class FraudReportPDF(FPDF):
    def header(self):
        # Logo / Title
        self.set_font("helvetica", "B", 20)
        self.set_text_color(8, 119, 166) # SPL_BLUE
        self.cell(0, 10, "FraudGuard Executive Report", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("helvetica", "I", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_fraud_report(fraud_count, legit_count, total_amt, metrics):
    pdf = FraudReportPDF()
    pdf.add_page()
    
    # KPIs
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Key Performance Indicators", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(50, 50, 50)
    
    fraud_pct = fraud_count / (fraud_count + legit_count) * 100 if (fraud_count + legit_count) > 0 else 0
    
    kpis = [
        f"Total Transactions Processed: {fraud_count + legit_count:,}",
        f"Verified Legitimate: {legit_count:,}",
        f"Confirmed Fraudulent: {fraud_count:,}",
        f"Fraud Rate: {fraud_pct:.3f}%",
        f"Total Processed Amount (Approx): INR {total_amt:,.2f}"
    ]
    
    for k in kpis:
        pdf.cell(10)
        pdf.cell(0, 8, k, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(5)
    
    # Model Integrity
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "2. Active Model Integrity (XGBoost)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(50, 50, 50)
    if metrics:
        model_stats = [
            f"Accuracy:  {metrics.get('accuracy', 0)*100:.2f}%",
            f"Precision: {metrics.get('precision', 0)*100:.2f}%",
            f"Recall:    {metrics.get('recall', 0)*100:.2f}%",
            f"F1-Score:  {metrics.get('f1_score', 0)*100:.2f}%",
            f"ROC-AUC:   {metrics.get('roc_auc', 0)*100:.2f}%"
        ]
        for s in model_stats:
            pdf.cell(10)
            pdf.cell(0, 8, s, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(10)
        pdf.cell(0, 8, "Model metrics unavailable.", new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    
    # Summary Insights
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "3. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 11)
    summary_text = (
        "This report was generated automatically by the FraudGuard AI Dashboard. "
        "The system runs continuous monitoring of incoming PCA-anonymised transaction signatures "
        "using ensemble Machine Learning algorithms (XGBoost). Ensure security teams review flagged "
        "transactions with SHAP-explained properties via the interactive portal."
    )
    pdf.multi_cell(0, 6, summary_text)
    
    # Ensure it returns bytes correctly for fpdf2
    return bytes(pdf.output())
