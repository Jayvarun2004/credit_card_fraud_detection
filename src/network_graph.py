"""
network_graph.py - Generate Fraud Link Analysis Plotly Network Graph
"""

import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Splunk-style constants
SPL_GREEN = "#53a051"
SPL_RED   = "#dc4e41"

def build_fraud_network(df, n_samples=150, threshold=2.5):
    """
    Builds a NetworkX graph from a dataframe.
    Edges are created if transactions are mathematically similar.
    Returns a Plotly Figure.
    """
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42)
    
    target_col = 'Class' if 'Class' in df.columns else 'prediction' if 'prediction' in df.columns else None
    
    if target_col is None:
        labels = [0] * len(df)
    else:
        labels = df[target_col].values
        
    features = [f'V{i}' for i in range(1, 29)]
    existing_feats = [f for f in features if f in df.columns]
    X = df[existing_feats].fillna(0).values
    
    dist_matrix = euclidean_distances(X, X)
    
    G = nx.Graph()
    amounts = df['Amount'].values if 'Amount' in df.columns else np.zeros(len(df))
    
    for i in range(len(df)):
        G.add_node(i, label=labels[i], amount=amounts[i])
        
    # Add edges if distance < threshold
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if dist_matrix[i, j] < threshold:
                G.add_edge(i, j, weight=round(float(dist_matrix[i, j]), 2))
                
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#4b5260'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        lab = G.nodes[node]['label']
        amt = G.nodes[node]['amount']
        
        c = SPL_RED if lab == 1 else SPL_GREEN
        node_colors.append(c)
        s = min(max(amt * 83 / 500.0, 6), 25) # Scale size by amount roughly
        node_sizes.append(s)
        
        status = "Fraud" if lab == 1 else "Legit"
        node_texts.append(f"Status: {status}<br>Amount: ₹{amt*83:.2f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_texts,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=1.5,
            line_color='#1a1c21'))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
