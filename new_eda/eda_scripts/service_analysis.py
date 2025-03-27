#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of medical services and service types
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
os.makedirs('eda_results/unfiltered/figures/service', exist_ok=True)

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

def service_type_analysis(df):
    """Analyze service types and their payment distributions"""
    print("\n=== SERVICE TYPE ANALYSIS ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Note: 'service_type_id' column not found, using 'service_code' as service type indicator")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Count services by type/code
    service_counts = df[service_type_column].value_counts().reset_index()
    service_counts.columns = [service_type_column, 'count']
    
    # Calculate payment totals by service type/code
    service_payments = df.groupby(service_type_column)['service_amount_net'].agg(['sum', 'mean', 'median', 'std']).reset_index()
    
    # Merge counts and payments
    service_stats = pd.merge(service_counts, service_payments, on=service_type_column)
    service_stats = service_stats.sort_values('sum', ascending=False)
    
    # Display top service types
    print("\nTop 10 Service Types by Total Payment:")
    print(service_stats.head(10).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    
    # Save service stats to file
    service_stats.to_csv('eda_results/unfiltered/data/service_type_stats.csv', index=False)
    
    # Plot top service types
    plt.figure(figsize=(14, 8))
    top_services = service_stats.head(10).copy()
    # Convert to millions for better display
    top_services['sum_millions'] = top_services['sum'] / 1e6
    plt.bar(top_services[service_type_column].astype(str), top_services['sum_millions'], 
            color=sns.color_palette('viridis', 10))
    plt.title('Top 10 Service Types by Total Payment', fontsize=18, fontweight='bold')
    plt.xlabel('Service Type', fontsize=14)
    plt.ylabel('Total Payment (Millions)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/service/top_service_types.png', dpi=300, bbox_inches='tight')
    
    # Analyze service type trends over time
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['service_date'].dt.strftime('%Y-%m')
    
    # Get top 5 service types
    top5_service_types = service_stats.head(5)[service_type_column].tolist()
    
    # Filter for top service types
    top_services_df = df_copy[df_copy[service_type_column].isin(top5_service_types)]
    
    # Group by month and service type
    monthly_by_service = top_services_df.groupby(['year_month', service_type_column])['service_amount_net'].sum().reset_index()
    
    # Create pivot table for plotting
    pivot_services = monthly_by_service.pivot(index='year_month', columns=service_type_column, values='service_amount_net')
    
    # Ensure datetime index for proper ordering
    pivot_services.index = pd.to_datetime(pivot_services.index)
    pivot_services = pivot_services.sort_index()
    
    # Plot time trends for top service types
    plt.figure(figsize=(16, 8))
    for service_type in top5_service_types:
        if service_type in pivot_services.columns:
            plt.plot(pivot_services.index, pivot_services[service_type]/1e6, marker='o', label=f'{service_type}')
    
    plt.title('Monthly Payments by Top 5 Service Types', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Total Payment (Millions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Service Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/service/service_type_trends.png', dpi=300, bbox_inches='tight')
    
    return service_stats

def service_growth_analysis(df):
    """Analyze growth patterns for different service types"""
    print("\n=== SERVICE GROWTH ANALYSIS ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Note: 'service_type_id' column not found, using 'service_code' as service type indicator")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
        
    # Add year-month field
    df['year'] = df['service_date'].dt.year
    
    # Calculate growth by service type between years
    years = sorted(df['year'].unique())
    if len(years) >= 2:
        # Get the two most recent years
        current_year = years[-1]
        previous_year = years[-2]
        
        # Group by service type and year
        service_by_year = df.groupby([service_type_column, 'year'])['service_amount_net'].sum().reset_index()
        
        # Create pivot table
        service_pivot = service_by_year.pivot(index=service_type_column, columns='year', values='service_amount_net')
        
        # Calculate growth rates
        if previous_year in service_pivot.columns and current_year in service_pivot.columns:
            service_pivot['growth_rate'] = (service_pivot[current_year] - service_pivot[previous_year]) / service_pivot[previous_year] * 100
            service_pivot['payment_diff'] = service_pivot[current_year] - service_pivot[previous_year]
            
            # Sort by absolute payment difference
            service_pivot_sorted = service_pivot.sort_values('payment_diff', ascending=False)
            
            # Display top growing and declining services
            print(f"\nTop 5 Growing Service Types ({previous_year} to {current_year}):")
            top_growing = service_pivot_sorted.head(5).reset_index()
            print(top_growing[[service_type_column, previous_year, current_year, 'growth_rate', 'payment_diff']].to_string(
                index=False, float_format=lambda x: f"{x:,.2f}"))
            
            print(f"\nTop 5 Declining Service Types ({previous_year} to {current_year}):")
            top_declining = service_pivot_sorted.tail(5).sort_values('payment_diff').reset_index()
            print(top_declining[[service_type_column, previous_year, current_year, 'growth_rate', 'payment_diff']].to_string(
                index=False, float_format=lambda x: f"{x:,.2f}"))
            
            # Save growth analysis to file
            service_pivot.reset_index().to_csv('eda_results/unfiltered/data/service_growth.csv', index=False)
            
            # Plot top 5 growing and declining services
            plt.figure(figsize=(14, 8))
            
            # Create bar positions
            top_services = list(top_growing[service_type_column].astype(str)) + list(top_declining[service_type_column].astype(str))
            x = np.arange(len(top_services))
            growth_rates = list(top_growing['growth_rate']) + list(top_declining['growth_rate'])
            
            # Set colors - positive growth in blue, negative in red
            colors = ['blue' if rate > 0 else 'red' for rate in growth_rates]
            
            # Plot bars
            plt.bar(x, growth_rates, color=colors)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.title(f'Service Types with Largest Growth Changes ({previous_year} to {current_year})', 
                     fontsize=18, fontweight='bold')
            plt.xlabel('Service Type', fontsize=14)
            plt.ylabel('Growth Rate (%)', fontsize=14)
            plt.xticks(x, top_services, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add growth rate labels
            for i, rate in enumerate(growth_rates):
                plt.text(i, rate + (5 if rate > 0 else -15), f"{rate:.1f}%", 
                         ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('eda_results/unfiltered/figures/service/service_growth.png', dpi=300, bbox_inches='tight')

def average_payment_analysis(df):
    """Analyze average payment per service over time"""
    print("\n=== AVERAGE PAYMENT ANALYSIS ===")
    
    # Group by month
    monthly_avg = df.groupby(pd.Grouper(key='service_date', freq='M')).agg(
        total_payment=('service_amount_net', 'sum'),
        service_count=('service_document_id', 'count'),
        unique_patients=('patient_id', 'nunique')
    ).reset_index()
    
    # Calculate average payment per service
    monthly_avg['avg_payment'] = monthly_avg['total_payment'] / monthly_avg['service_count']
    monthly_avg['avg_payment_per_patient'] = monthly_avg['total_payment'] / monthly_avg['unique_patients']
    
    # Save average payment data
    monthly_avg.to_csv('eda_results/unfiltered/data/monthly_avg_payments.csv', index=False)
    
    # Plot average payment trends
    plt.figure(figsize=(14, 8))
    
    plt.plot(monthly_avg['service_date'], monthly_avg['avg_payment'], 
             marker='o', linestyle='-', linewidth=2, label='Avg Payment per Service')
    
    # Add trend line
    x = np.arange(len(monthly_avg))
    z = np.polyfit(x, monthly_avg['avg_payment'], 1)
    p = np.poly1d(z)
    plt.plot(monthly_avg['service_date'], p(x), linestyle='--', color='red', alpha=0.7, label='Trend')
    
    plt.title('Average Payment per Service Over Time', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Payment (₽)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/service/avg_payment_trend.png', dpi=300, bbox_inches='tight')
    
    # Plot metrics comparison
    plt.figure(figsize=(14, 8))
    
    # Create second y-axis for count
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Plot average payment on first axis
    ax1.plot(monthly_avg['service_date'], monthly_avg['avg_payment'], 
             marker='o', color='blue', label='Avg Payment per Service')
    ax1.set_ylabel('Average Payment (₽)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot service count on second axis
    ax2.plot(monthly_avg['service_date'], monthly_avg['service_count'], 
             marker='s', color='green', label='Service Count')
    ax2.set_ylabel('Service Count', color='green', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add title and formatting
    plt.title('Average Payment vs Service Count', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/service/avg_payment_vs_count.png', dpi=300, bbox_inches='tight')
    
    # Plot payment per patient
    plt.figure(figsize=(14, 8))
    plt.plot(monthly_avg['service_date'], monthly_avg['avg_payment_per_patient'], 
             marker='o', linestyle='-', linewidth=2, color='purple')
    
    # Add trend line
    z2 = np.polyfit(x, monthly_avg['avg_payment_per_patient'], 1)
    p2 = np.poly1d(z2)
    plt.plot(monthly_avg['service_date'], p2(x), linestyle='--', color='red', alpha=0.7)
    
    plt.title('Average Payment per Patient Over Time', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Payment per Patient (₽)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/patient/avg_payment_per_patient.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Load processed data
    df = load_processed_data()
    
    if df is not None:
        # Perform service type analysis
        service_stats = service_type_analysis(df)
        
        # Analyze service growth
        service_growth_analysis(df)
        
        # Analyze average payments
        average_payment_analysis(df)
        
        print("\nService analysis completed successfully!") 