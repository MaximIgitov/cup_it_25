#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive EDA for medical services data
WITHOUT filtering or removing duplicates/outliers
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# For better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('data/unfiltered', exist_ok=True)
os.makedirs('eda_results/unfiltered/figures', exist_ok=True)

def load_data(filepath='CupIT_Sber_data.csv'):
    """Load and prepare the dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    
    # Display basic info
    print(f"\nDataset shape: {df.shape}")
    
    return df

def preprocess_data_minimal(df):
    """Minimal preprocessing - just convert types without filtering"""
    print("\n=== MINIMAL PREPROCESSING (NO FILTERING) ===")
    
    # Convert service_date to datetime
    df['service_date'] = pd.to_datetime(df['service_date'])
    
    # Extract date components
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    df['day'] = df['service_date'].dt.day
    df['dayofweek'] = df['service_date'].dt.dayofweek
    
    # Convert service_amount_net to numeric (keep all values)
    if df['service_amount_net'].dtype == 'object':
        print("Converting service_amount_net from string to numeric...")
        # Replace commas with periods first (if needed)
        df['service_amount_net'] = df['service_amount_net'].astype(str).str.replace(',', '.')
        df['service_amount_net'] = pd.to_numeric(df['service_amount_net'], errors='coerce')
    
    # DON'T drop any rows with null values
    # DON'T remove outliers
    # DON'T deduplicate
    
    # Just replace any remaining NaN with 0 for numeric fields
    df['service_amount_net'] = df['service_amount_net'].fillna(0)
    
    print(f"Processed data shape: {df.shape} (no records removed)")
    return df

def save_processed_data(df):
    """Save processed data for further analysis"""
    df.to_csv('data/unfiltered/processed_data.csv', index=False)
    print("\nProcessed data saved to 'data/unfiltered/processed_data.csv'")

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Minimal preprocessing - keep all data
    data = preprocess_data_minimal(data)
    
    # Save unfiltered processed data
    save_processed_data(data)
    
    print("\nPreprocessing without filtering completed successfully!") 