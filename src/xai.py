"""
xai.py - Explainable AI Logic for Individual Transactions
"""
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go

SPL_GREEN = "#53a051"
SPL_RED   = "#dc4e41"

def extract_shap_values(model, feature_names, feature_df):
    """
    Given a model (e.g., XGBoost) and a single row DataFrame,
    returns the expected value and shap values.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        expected_value = explainer.expected_value
        
        # Binary classification handling
        if isinstance(expected_value, (np.ndarray, list)):
            expected_value = expected_value[1] 
        if isinstance(shap_values, list):
            shap_values = shap_values[1] 
            
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        return expected_value, shap_values
    except Exception as e:
        return None, None

def get_shap_waterfall_plotly(model, feature_df, title="Prediction Explanation (SHAP)"):
    """
    Generates a Plotly Waterfall chart for SHAP values.
    """
    feature_names = feature_df.columns.tolist()
    expected_value, shap_values = extract_shap_values(model, feature_names, feature_df)
    
    if shap_values is None:
        fig = go.Figure()
        fig.add_annotation(text="XAI Explainer not supported for this model.", showarrow=False, font=dict(color="#c3cbd4"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    impacts = np.abs(shap_values)
    
    # Top 10 indices
    top_indices = np.argsort(impacts)[-10:]
    
    keep_shap = shap_values[top_indices]
    keep_names = [feature_names[i] for i in top_indices]
    keep_vals = feature_df.iloc[0].values[top_indices]
    
    mask = np.ones(len(shap_values), dtype=bool)
    mask[top_indices] = False
    rest_sum = np.sum(shap_values[mask])
    
    measure = ["absolute"] + ["relative"] * (len(keep_shap) + 1) + ["total"]
    
    x_labels = ["Base Value"] + ["Other Features"] + [f"{f} ({v:.2f})" for f, v in zip(keep_names, keep_vals)] + ["Final Score"]
    y_values = [expected_value, rest_sum] + keep_shap.tolist() + [0]
    
    text_vals = [f"{expected_value:.2f}", f"{rest_sum:+.2f}"] + [f"{v:+.2f}" for v in keep_shap] + [f"{sum(y_values[:-1]):.2f}"]
    
    fig = go.Figure(go.Waterfall(
        name = "SHAP",
        orientation = "v",
        measure = measure,
        x = x_labels,
        textposition = "outside",
        text = text_vals,
        y = y_values,
        connector = {"line":{"color":"#2d3040"}},
        decreasing = {"marker":{"color":SPL_GREEN}},
        increasing = {"marker":{"color":SPL_RED}},
        totals = {"marker":{"color":"#0877a6"}}
    ))
    
    fig.update_layout(
        title = dict(text=title, font=dict(color="#c3cbd4", size=12)),
        waterfallgap = 0.3,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c3cbd4", family="Roboto", size=10),
        xaxis=dict(gridcolor="#2d3040", linecolor="#2d3040", tickangle=-45),
        yaxis=dict(gridcolor="#2d3040", linecolor="#2d3040", title="Risk Log-Odds Impact"),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    
    return fig
