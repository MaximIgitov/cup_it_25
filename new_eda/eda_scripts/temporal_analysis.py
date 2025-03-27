#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal analysis of payment data without filtering
This script analyzes payment trends over time, providing insights into monthly and yearly patterns
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
os.makedirs('eda_results/unfiltered/figures/temporal', exist_ok=True)

def load_unfiltered_data(filepath='data/unfiltered/processed_data.csv'):
    """Load the unfiltered processed dataset"""
    print(f"Loading unfiltered data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['service_date'] = pd.to_datetime(df['service_date'])
        return df
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        print("Please run eda_scripts/eda_main.py first.")
        return None

def monthly_payment_analysis(df):
    """Analyze monthly payment trends and year-over-year growth"""
    print("\n=== MONTHLY PAYMENT ANALYSIS (UNFILTERED DATA) ===")
    
    # Group by month
    monthly_data = df.groupby(pd.Grouper(key='service_date', freq='M'))['service_amount_net'].sum().reset_index()
    monthly_data['year_month'] = monthly_data['service_date'].dt.strftime('%Y-%m')
    monthly_data['year'] = monthly_data['service_date'].dt.year
    monthly_data['month'] = monthly_data['service_date'].dt.month
    
    # Save monthly aggregated data
    monthly_data.to_csv('eda_results/unfiltered/data/monthly_aggregated_payments.csv', index=False)
    
    # Print some insights
    first_month = monthly_data.iloc[0]
    last_month = monthly_data.iloc[-1]
    print(f"First month: {first_month['year_month']}, amount: {first_month['service_amount_net']:,.2f}")
    print(f"Last month: {last_month['year_month']}, amount: {last_month['service_amount_net']:,.2f}")
    
    # Calculate yearly totals
    yearly_totals = monthly_data.groupby('year')['service_amount_net'].sum()
    
    print("\nYearly Totals (UNFILTERED DATA):")
    for year, total in yearly_totals.items():
        print(f"Year {year}: {total:,.2f}")
    
    # Calculate growth if we have multiple years
    years = sorted(monthly_data['year'].unique())
    if len(years) >= 2:
        current_year = years[-1]
        previous_year = years[-2]
        growth = (yearly_totals[current_year] - yearly_totals[previous_year]) / yearly_totals[previous_year] * 100
        print(f"Growth {previous_year} to {current_year}: {growth:.2f}%")
    
    # Year-over-year growth by month
    print("\nYear-over-Year Growth by Month (UNFILTERED DATA):\n")
    
    # Create pivot table for month-by-month comparison
    pivot_data = monthly_data.pivot_table(index='month', columns='year', values='service_amount_net')
    
    # Debug information
    print("Debug - Raw values from pivot table (UNFILTERED DATA):")
    months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    for month_num, month_name in months.items():
        if month_num in pivot_data.index:
            values = []
            for year in years:
                if year in pivot_data.columns:
                    value = pivot_data.loc[month_num, year]
                    values.append(f"{year}={value:.2f}")
            print(f"  {month_name}: {', '.join(values)}")
    
    # Calculate and display YoY growth
    if len(years) >= 2:
        print(f"\nComparing {previous_year} to {current_year} (UNFILTERED DATA):")
        for month_num, month_name in months.items():
            if month_num in pivot_data.index and previous_year in pivot_data.columns and current_year in pivot_data.columns:
                if not pd.isna(pivot_data.loc[month_num, previous_year]) and pivot_data.loc[month_num, previous_year] != 0:
                    yoy_growth = (pivot_data.loc[month_num, current_year] - pivot_data.loc[month_num, previous_year]) / pivot_data.loc[month_num, previous_year] * 100
                    print(f"  {month_name}: {yoy_growth:.2f}%")
    
    # Year-over-year comparison visualization
    plot_year_over_year_comparison(pivot_data)
    
    return monthly_data

def plot_year_over_year_comparison(pivot_data):
    """Create a visualized year-over-year comparison"""
    
    # Print debugging information
    print(f"\nYears in pivot_data: {pivot_data.columns.tolist()}")
    
    # Output raw values for debugging
    print("\nDebug - Pivot table raw values:")
    print(pivot_data)
    
    # Plot the data
    plt.figure(figsize=(14, 8))
    
    # Define a list of months for X-axis
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Get years from pivot table
    years = pivot_data.columns.tolist()
    print(f"\nGenerating Year-over-Year chart with years: {years}")
    
    # Plot each year as a line
    for year in years:
        data_to_plot = []
        for month in range(1, 13):
            if month in pivot_data.index and year in pivot_data.columns:
                value = pivot_data.loc[month, year] / 1e6  # Convert to millions
                data_to_plot.append(value)
                print(f"  Month {month}: {value:.2f}M")
            else:
                data_to_plot.append(0)
        
        plt.plot(range(1, 13), data_to_plot, marker='o', linewidth=2, label=str(year))
    
    plt.title('Year-over-Year Payment Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Total Payment (Millions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Year')
    plt.xticks(range(1, 13), months)
    
    # Add value labels
    for year in years:
        for month in range(1, 13):
            if month in pivot_data.index and year in pivot_data.columns:
                value = pivot_data.loc[month, year] / 1e6
                plt.text(month, value + 2, f"{value:.1f}M", ha='center')
    
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/temporal/year_over_year_comparison.png', dpi=300, bbox_inches='tight')
    
    # If we have at least 2 years, also plot YoY growth
    if len(years) >= 2:
        current_year = years[-1]
        previous_year = years[-2]
        
        plt.figure(figsize=(14, 8))
        
        # Calculate growth rate
        growths = []
        for month in range(1, 13):
            if month in pivot_data.index and previous_year in pivot_data.columns and current_year in pivot_data.columns:
                prev_value = pivot_data.loc[month, previous_year]
                curr_value = pivot_data.loc[month, current_year]
                if prev_value > 0:  # Avoid division by zero
                    growth = ((curr_value - prev_value) / prev_value) * 100
                    growths.append(growth)
                else:
                    growths.append(0)
            else:
                growths.append(0)
        
        # Create bar chart of growth rates
        plt.bar(range(1, 13), growths, color=sns.color_palette('viridis', 12))
        
        plt.title(f'Monthly Payment Growth ({previous_year} to {current_year})', fontsize=18, fontweight='bold')
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Growth Rate (%)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(range(1, 13), months)
        
        # Add growth rate labels
        for i, growth in enumerate(growths):
            plt.text(i + 1, growth + (2 if growth >= 0 else -5), f"{growth:.1f}%", ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_results/unfiltered/figures/temporal/yoy_growth.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Load unfiltered data
    df = load_unfiltered_data()
    
    if df is not None:
        # Process monthly payment analysis
        monthly_data = monthly_payment_analysis(df)
        
        print("\nTemporal analysis with unfiltered data completed successfully!") 