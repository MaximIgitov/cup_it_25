#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of patient data and demographics
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

# Create necessary directories
os.makedirs('eda_results/unfiltered/data', exist_ok=True)
os.makedirs('eda_results/unfiltered/figures/patient', exist_ok=True)

def load_processed_data(filepath='data/unfiltered/processed_data.csv'):
    """Load the processed dataset"""
    print(f"Loading processed data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['service_date'] = pd.to_datetime(df['service_date'])
        return df
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        print("Please run eda_scripts/eda_main.py first.")
        return None

def patient_visit_analysis(df):
    """Analyze patient visit patterns"""
    print("\n=== PATIENT VISIT ANALYSIS ===")
    
    # Count visits per patient
    visits_per_patient = df.groupby('patient_id')['service_document_id'].nunique().reset_index()
    visits_per_patient.columns = ['patient_id', 'visit_count']
    
    # Calculate summary statistics
    visit_stats = visits_per_patient['visit_count'].describe()
    
    print("\nPatient Visit Statistics:")
    print(f"Total patients: {len(visits_per_patient):,}")
    print(f"Average visits per patient: {visit_stats['mean']:.2f}")
    print(f"Median visits per patient: {visit_stats['50%']:.2f}")
    print(f"Max visits from a single patient: {visit_stats['max']:.0f}")
    
    # Save patient visit stats
    visits_per_patient.to_csv('eda_results/unfiltered/data/patient_visit_stats.csv', index=False)
    
    # Plot visit distribution
    plt.figure(figsize=(14, 8))
    
    # Filter to reasonable range for better visualization
    plotting_data = visits_per_patient[visits_per_patient['visit_count'] <= 20]  # Focus on patients with ≤ 20 visits
    
    # Get visit counts
    visit_counts = plotting_data['visit_count'].value_counts().sort_index()
    
    # Plot as bar chart
    plt.bar(visit_counts.index, visit_counts.values, color=sns.color_palette('viridis', len(visit_counts)))
    
    plt.title('Patient Visit Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Number of Visits', fontsize=14)
    plt.ylabel('Number of Patients', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(np.arange(1, 21))
    
    # Add count labels on bars
    for i, count in enumerate(visit_counts.values):
        plt.text(visit_counts.index[i], count + (count * 0.02), 
                 f"{count:,}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/patient_visit_distribution.png', dpi=300, bbox_inches='tight')
    
    return visits_per_patient

def patient_retention_analysis(df):
    """Analyze patient retention over time"""
    print("\n=== PATIENT RETENTION ANALYSIS ===")
    
    # Extract year-month and add to dataframe
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Get unique patients per month
    monthly_patients = df.groupby('year_month')['patient_id'].nunique().reset_index()
    monthly_patients.columns = ['year_month', 'unique_patients']
    
    # Sort by year-month
    monthly_patients['date'] = pd.to_datetime(monthly_patients['year_month'] + '-01')
    monthly_patients = monthly_patients.sort_values('date')
    
    # Calculate new vs returning patients
    all_patients = set()
    new_patients = []
    returning_patients = []
    
    for _, row in monthly_patients.iterrows():
        ym = row['year_month']
        month_patients = set(df[df['year_month'] == ym]['patient_id'].unique())
        
        # Calculate new patients
        new = month_patients - all_patients
        new_patients.append(len(new))
        
        # Calculate returning patients
        returning = month_patients & all_patients
        returning_patients.append(len(returning))
        
        # Update all_patients
        all_patients.update(month_patients)
    
    # Add calculated values to dataframe
    monthly_patients['new_patients'] = new_patients
    monthly_patients['returning_patients'] = returning_patients
    
    # Save retention data
    monthly_patients.to_csv('eda_results/unfiltered/data/patient_retention.csv', index=False)
    
    # Plot new vs returning patients
    plt.figure(figsize=(14, 8))
    
    # Create stacked bar chart
    plt.bar(range(len(monthly_patients)), monthly_patients['new_patients'], label='New Patients',
            color='#1f77b4')
    plt.bar(range(len(monthly_patients)), monthly_patients['returning_patients'], 
            bottom=monthly_patients['new_patients'], label='Returning Patients',
            color='#ff7f0e')
    
    plt.title('Patient Retention: New vs Returning Patients', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Number of Patients', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # X-axis labels (use every other month for clarity)
    step = 2 if len(monthly_patients) > 12 else 1
    plt.xticks(range(0, len(monthly_patients), step), 
              [m for i, m in enumerate(monthly_patients['year_month']) if i % step == 0],
              rotation=45)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/patient_retention.png', dpi=300, bbox_inches='tight')
    
    # Calculate and plot retention rate
    monthly_patients['retention_rate'] = monthly_patients['returning_patients'] / monthly_patients['unique_patients'] * 100
    
    plt.figure(figsize=(14, 8))
    plt.plot(range(len(monthly_patients)), monthly_patients['retention_rate'], 
            marker='o', linestyle='-', linewidth=2)
    
    plt.title('Monthly Patient Retention Rate', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Retention Rate (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # X-axis labels (use every other month for clarity)
    plt.xticks(range(0, len(monthly_patients), step), 
              [m for i, m in enumerate(monthly_patients['year_month']) if i % step == 0],
              rotation=45)
    
    # Add percentage labels
    for i, rate in enumerate(monthly_patients['retention_rate']):
        plt.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/retention_rate.png', dpi=300, bbox_inches='tight')
    
    return monthly_patients

def service_mix_analysis(df):
    """Analyze service type mix by patient segment"""
    print("\n=== SERVICE MIX ANALYSIS ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Note: 'service_type_id' column not found, using 'service_code' as service type indicator")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Group patients by visit frequency
    visits_per_patient = df.groupby('patient_id')['service_document_id'].nunique().reset_index()
    visits_per_patient.columns = ['patient_id', 'visit_count']
    
    # Define patient segments
    visits_per_patient['segment'] = pd.cut(
        visits_per_patient['visit_count'], 
        bins=[0, 1, 3, 10, float('inf')],
        labels=['One-time', 'Occasional', 'Regular', 'Power']
    )
    
    # Merge segment info back to df
    df_with_segments = pd.merge(df, visits_per_patient[['patient_id', 'segment']], on='patient_id')
    
    # Get service type distribution by segment
    segment_service_mix = df_with_segments.groupby(['segment', service_type_column]).size().unstack().fillna(0)
    
    # Convert to percentages
    segment_service_pct = segment_service_mix.div(segment_service_mix.sum(axis=1), axis=0) * 100
    
    # Save service mix data
    segment_service_pct.to_csv('eda_results/unfiltered/data/segment_service_mix.csv')
    
    # Get top 5 services for visualization
    top_services = df[service_type_column].value_counts().nlargest(5).index.tolist()
    
    # Filter for top services
    service_mix_plot = segment_service_pct[top_services].copy()
    
    # Plot service mix by segment
    plt.figure(figsize=(14, 8))
    service_mix_plot.plot(kind='bar', stacked=False, ax=plt.gca(),
                         color=sns.color_palette('viridis', len(top_services)))
    
    plt.title('Service Mix by Patient Segment', fontsize=18, fontweight='bold')
    plt.xlabel('Patient Segment', fontsize=14)
    plt.ylabel('Service Usage (%)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title='Service Type')
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/segment_service_mix.png', dpi=300, bbox_inches='tight')
    
    # Calculate average payment per segment
    segment_payment = df_with_segments.groupby('segment')['service_amount_net'].agg(['sum', 'mean']).reset_index()
    
    # Plot average payment per segment
    plt.figure(figsize=(14, 8))
    plt.bar(segment_payment['segment'], segment_payment['mean'],
           color=sns.color_palette('viridis', len(segment_payment)))
    
    plt.title('Average Payment per Visit by Patient Segment', fontsize=18, fontweight='bold')
    plt.xlabel('Patient Segment', fontsize=14)
    plt.ylabel('Average Payment (₽)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add payment labels
    for i, payment in enumerate(segment_payment['mean']):
        plt.text(i, payment + (payment * 0.03), f"{payment:,.0f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/segment_payment.png', dpi=300, bbox_inches='tight')
    
    # Save segment payment data
    segment_payment.to_csv('eda_results/unfiltered/data/segment_payment.csv', index=False)
    
    return segment_service_pct, segment_payment

if __name__ == "__main__":
    # Load processed data
    df = load_processed_data()
    
    if df is not None:
        # Analyze patient visits
        visits_per_patient = patient_visit_analysis(df)
        
        # Analyze patient retention
        monthly_patients = patient_retention_analysis(df)
        
        # Analyze service mix by patient segment
        service_mix, segment_payment = service_mix_analysis(df)
        
        print("\nPatient analysis completed successfully!") 