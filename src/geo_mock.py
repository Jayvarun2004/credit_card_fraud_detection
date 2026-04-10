"""
geo_mock.py - Deterministic Lat/Lon & Metadata mapping for anonymized PCA features.
"""

import numpy as np
import pandas as pd
import hashlib

def mock_ip_and_geo(row):
    """
    Generate synthetic deterministic geo-data from V1, V2, Amount to simulate IP and Location.
    Uses MD5 hash of V1, V2 strings to get a consistent mapping.
    """
    v1 = row.get('V1', 0.0)
    v2 = row.get('V2', 0.0)
    amt = row.get('Amount', 0.0)
    
    # Hash to get pseudo-random deterministic bytes
    h = hashlib.md5(f"{v1}_{v2}_{amt}".encode('utf-8')).digest()
    
    # Map to Lat: -90 to 90
    lat = -90 + (h[0] / 255.0) * 180
    # Map to Lon: -180 to 180
    lon = -180 + (h[1] / 255.0) * 360
    
    if row.get('Class', 0) == 1 or row.get('prediction', 0) == 1:
        # Fraud clusters - pull them towards specific hotspots deterministically
        hotspots = [
            (35.86, 104.19), # China
            (55.37, -3.43),  # UK
            (51.52, 46.01),  # Russia
            (-23.55, -46.63),# Brazil
            (37.09, -95.71)  # USA
        ]
        hs = hotspots[h[2] % len(hotspots)]
        # Add some jitter around the hotspot
        lat = hs[0] + (h[3] / 255.0 - 0.5) * 15
        lon = hs[1] + (h[4] / 255.0 - 0.5) * 15
    
    # Restrict lat/lon to valid ranges
    lat = max(-90, min(90, lat))
    lon = max(-180, min(180, lon))
    
    return pd.Series([lat, lon, f"192.168.{h[5]}.{h[6]}"])

def add_geo_features(df):
    """Appends 'Lat', 'Lon', 'IP' to the dataframe."""
    df_out = df.copy()
    if len(df_out) == 0:
        return df_out
    
    geo_df = df_out.apply(mock_ip_and_geo, axis=1)
    geo_df.columns = ['Lat', 'Lon', 'IP']
    
    return pd.concat([df_out, geo_df], axis=1)
