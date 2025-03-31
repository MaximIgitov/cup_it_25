#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of insurance services and their types
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from matplotlib.ticker import FuncFormatter

# Import custom styling
import eda_style

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
    print("\n=== АНАЛИЗ ТИПОВ СТРАХОВЫХ УСЛУГ ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Note: 'service_type_id' column not found, using 'service_code' as service type indicator")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Count services by type
    service_counts = df[service_type_column].value_counts().reset_index()
    service_counts.columns = ['service_type', 'count']
    
    # Calculate payment totals by service type
    service_payments = df.groupby(service_type_column)['service_amount_net'].agg(['sum', 'mean', 'count']).reset_index()
    service_payments.columns = ['service_type', 'total_payment', 'avg_payment', 'count']
    
    # Sort by total revenue
    service_payments = service_payments.sort_values('total_payment', ascending=False)
    
    # Calculate percent of total revenue
    total_revenue = service_payments['total_payment'].sum()
    service_payments['pct_of_total'] = service_payments['total_payment'] / total_revenue * 100
    
    # Add cumulative percentage
    service_payments['cumulative_pct'] = service_payments['pct_of_total'].cumsum()
    
    # Save service payments data
    service_payments.to_csv('eda_results/unfiltered/data/service_payments.csv', index=False)
    
    # Print summary
    print("\nРаспределение по типам услуг:")
    print(f"Всего уникальных типов услуг: {len(service_payments)}")
    print(f"Топ-5 типов услуг по объему выплат:")
    
    for _, row in service_payments.head(5).iterrows():
        print(f"  - Услуга {row['service_type']}: {row['total_payment']:,.0f} руб. ({row['pct_of_total']:.1f}%)")
    
    # Plot top 10 services by revenue
    top_10 = service_payments.head(10).copy()
    
    # Create figure for top services by revenue
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot horizontal bar chart with blue gradient colors
    bars = ax.barh(range(len(top_10)), top_10['total_payment'], 
                  color=[eda_style.COLORS['blue_grad_%d' % i] for i in range(10)])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Топ-10 услуг по объему выплат',
        xlabel='Объем выплат (руб.)',
        ylabel='Тип услуги'
    )
    
    # Set y-ticks
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['service_type'])
    
    # Format x-axis with currency
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000000:.1f}M руб."))
    
    # Add value labels
    for i, v in enumerate(top_10['total_payment']):
        ax.text(v + (v * 0.01), i, f"{v/1000000:.1f}M руб.", fontsize=14, va='center')
    
    # Save figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/top_services_revenue.png')
    
    # Create Pareto chart (80/20 analysis)
    fig, ax1 = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot bars for revenue percentage
    bars = ax1.bar(range(min(20, len(service_payments))), 
                  service_payments['pct_of_total'].head(20),
                  color=eda_style.COLORS['blue'])
    
    # Create second y-axis for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(range(min(20, len(service_payments))), 
             service_payments['cumulative_pct'].head(20), 
             marker='o', linewidth=3, markersize=8, 
             color=eda_style.COLORS['green'])
    
    # Draw 80% reference line
    ax2.axhline(y=80, color=eda_style.COLORS['cyan'], linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(len(service_payments.head(20))/2, 82, '80% выплат', 
             color=eda_style.COLORS['cyan'], fontsize=14, fontweight='bold')
    
    # Apply styles to first axis
    eda_style.apply_common_styles(
        ax1,
        title='Парето-анализ: вклад типов услуг в общий объем выплат',
        xlabel='Ранг типа услуги',
        ylabel='Процент от общего объема выплат (%)'
    )
    
    # Configure second axis
    ax2.set_ylabel('Накопленный процент (%)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(False)
    
    # Format y-axes
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
    
    # Set x-ticks
    ax1.set_xticks(range(min(20, len(service_payments))))
    ax1.set_xticklabels(service_payments['service_type'].head(20), rotation=90)
    
    # Save figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/pareto_service_revenue.png')
    
    return service_payments

def service_growth_analysis(df):
    """Analyze service growth over time"""
    print("\n=== АНАЛИЗ ДИНАМИКИ СТРАХОВЫХ УСЛУГ ===")
    
    # Extract year-month
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Group by year-month and service type
    monthly_services = df.groupby(['year_month', service_type_column]).size().reset_index()
    monthly_services.columns = ['year_month', 'service_type', 'count']
    
    # Add date column for sorting
    monthly_services['date'] = pd.to_datetime(monthly_services['year_month'] + '-01')
    monthly_services = monthly_services.sort_values('date')
    
    # Get top 5 services by overall count
    top_services = df[service_type_column].value_counts().nlargest(5).index.tolist()
    
    # Filter for these top services
    top_services_monthly = monthly_services[monthly_services['service_type'].isin(top_services)]
    
    # Pivot for plotting
    pivot_data = top_services_monthly.pivot(index='year_month', columns='service_type', values='count')
    
    # Save monthly service counts
    pivot_data.to_csv('eda_results/unfiltered/data/monthly_service_counts.csv')
    
    # Plot service trends
    fig, ax = eda_style.create_custom_figure(figsize=(16, 9))
    
    # Define colors for consistent visualization within blue-green-yellow spectrum
    colors = [eda_style.COLORS['blue'], eda_style.COLORS['cyan'], 
              eda_style.COLORS['green'], eda_style.COLORS['lime'], 
              eda_style.COLORS['yellow']]
    
    # Plot each service type
    for i, service in enumerate(pivot_data.columns):
        ax.plot(range(len(pivot_data)), pivot_data[service], 
                marker='o', linestyle='-', linewidth=3, markersize=8,
                label=f'Услуга {service}', color=colors[i % len(colors)])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячная динамика топ-5 типов страховых услуг',
        xlabel='Месяц',
        ylabel='Количество страховых случаев'
    )
    
    # X-axis labels - use every other month for clarity
    step = 2 if len(pivot_data) > 12 else 1
    ax.set_xticks(range(0, len(pivot_data), step))
    ax.set_xticklabels([m for i, m in enumerate(pivot_data.index) if i % step == 0],
                     rotation=45, ha='right')
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Add legend with better positioning
    ax.legend(fontsize=14, loc='upper left')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/service_trends.png')
    
    # Calculate growth rates (year-over-year if possible, otherwise month-over-month)
    # First, need to check if we have at least a year of data
    
    all_months = pd.Series(monthly_services['year_month'].unique()).sort_values()
    first_year = all_months.str[:4].min()
    last_year = all_months.str[:4].max()
    
    if first_year != last_year:  # We have multiple years
        print("\nРасчет темпов роста год к году...")
        
        # For each service type, calculate YoY growth
        growth_data = []
        
        for service in top_services:
            service_data = monthly_services[monthly_services['service_type'] == service]
            
            # Get months that appear in both years
            months_in_first = service_data[service_data['year_month'].str.startswith(first_year)]['year_month'].str[5:7]
            months_in_last = service_data[service_data['year_month'].str.startswith(last_year)]['year_month'].str[5:7]
            
            common_months = set(months_in_first).intersection(set(months_in_last))
            
            for month in common_months:
                prev_count = service_data[(service_data['year_month'] == f"{first_year}-{month}")]['count'].values[0]
                curr_count = service_data[(service_data['year_month'] == f"{last_year}-{month}")]['count'].values[0]
                
                growth_pct = ((curr_count - prev_count) / prev_count) * 100 if prev_count > 0 else float('inf')
                
                growth_data.append({
                    'service_type': service,
                    'month': month,
                    'prev_year_count': prev_count,
                    'curr_year_count': curr_count,
                    'growth_pct': growth_pct
                })
        
        growth_df = pd.DataFrame(growth_data)
        
        # Save growth data
        growth_df.to_csv('eda_results/unfiltered/data/service_growth.csv', index=False)
        
        # Plot growth rates
        fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
        
        # Create bar chart for each service's growth rate
        for i, service in enumerate(growth_df['service_type'].unique()):
            service_growth = growth_df[growth_df['service_type'] == service].copy()
            service_growth['month_name'] = pd.to_datetime('2020-' + service_growth['month'] + '-01').dt.strftime('%b')
            service_growth = service_growth.sort_values('month')
            
            x_positions = np.arange(len(service_growth)) + (i * 0.15)
            ax.bar(x_positions, service_growth['growth_pct'], width=0.15, 
                  label=f'Услуга {service}', color=colors[i % len(colors)])
        
        # Apply common styles
        eda_style.apply_common_styles(
            ax,
            title=f'Темп роста услуг год к году ({first_year} к {last_year})',
            xlabel='Месяц',
            ylabel='Темп роста (%)'
        )
        
        # Set x-ticks at the center of grouped bars
        month_names = growth_df.sort_values('month')['month'].unique()
        month_labels = pd.to_datetime('2020-' + pd.Series(month_names) + '-01').dt.strftime('%b')
        
        group_centers = np.arange(len(month_names)) + ((len(growth_df['service_type'].unique()) - 1) * 0.15 / 2)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(month_labels)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
        
        # Add zero reference line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=14, loc='best')
        
        # Save the figure
        eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/service_growth_yoy.png')
    
    return monthly_services

def average_payment_analysis(df):
    """Analyze average payment trends for services"""
    print("\n=== АНАЛИЗ СРЕДНИХ СТРАХОВЫХ ВЫПЛАТ ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Extract year-month
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Group by year-month and calculate average payments
    monthly_avg = df.groupby('year_month')['service_amount_net'].mean().reset_index()
    monthly_avg.columns = ['year_month', 'avg_payment']
    
    # Add date column for sorting
    monthly_avg['date'] = pd.to_datetime(monthly_avg['year_month'] + '-01')
    monthly_avg = monthly_avg.sort_values('date')
    
    # Save monthly average payments
    monthly_avg.to_csv('eda_results/unfiltered/data/monthly_avg_payments.csv', index=False)
    
    # Plot average payment trend
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot line chart
    ax.plot(range(len(monthly_avg)), monthly_avg['avg_payment'], 
           marker='o', linestyle='-', linewidth=3, markersize=8, color=eda_style.COLORS['cyan'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Динамика средней страховой выплаты по месяцам',
        xlabel='Месяц',
        ylabel='Средняя выплата (руб.)'
    )
    
    # X-axis labels - use every other month for clarity
    step = 2 if len(monthly_avg) > 12 else 1
    ax.set_xticks(range(0, len(monthly_avg), step))
    ax.set_xticklabels([m for i, m in enumerate(monthly_avg['year_month']) if i % step == 0],
                     rotation=45, ha='right')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Add average payment labels
    for i, payment in enumerate(monthly_avg['avg_payment']):
        if i % step == 0:  # Label every 'step' points for clarity
            ax.text(i, payment + (payment * 0.03), f"{payment:,.0f} руб.", 
                   ha='center', fontsize=14, fontweight='bold')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/avg_payment_trend.png')
    
    # Group by service type and calculate average payments
    service_avg = df.groupby(service_type_column)['service_amount_net'].mean().reset_index()
    service_avg.columns = ['service_type', 'avg_payment']
    
    # Sort by average payment
    service_avg = service_avg.sort_values('avg_payment', ascending=False)
    
    # Save service average payments
    service_avg.to_csv('eda_results/unfiltered/data/service_avg_payments.csv', index=False)
    
    # Plot top 10 services by average payment
    top_10_avg = service_avg.head(10).copy()
    
    # Create figure for top services by average payment
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot horizontal bar chart with blue gradient colors
    bars = ax.barh(range(len(top_10_avg)), top_10_avg['avg_payment'], 
                 color=[eda_style.COLORS['blue_grad_%d' % i] for i in range(10)])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Топ-10 услуг по среднему размеру выплаты',
        xlabel='Средняя выплата (руб.)',
        ylabel='Тип услуги'
    )
    
    # Set y-ticks
    ax.set_yticks(range(len(top_10_avg)))
    ax.set_yticklabels(top_10_avg['service_type'])
    
    # Format x-axis with currency
    ax.xaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Add value labels
    for i, v in enumerate(top_10_avg['avg_payment']):
        ax.text(v + (v * 0.01), i, f"{v:,.0f} руб.", fontsize=14, va='center')
    
    # Save figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/service/top_services_avg_payment.png')
    
    return monthly_avg, service_avg

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data()
    
    if data is not None:
        # Analyze service types
        service_payments = service_type_analysis(data)
        
        # Analyze service growth
        monthly_services = service_growth_analysis(data)
        
        # Analyze average payments
        monthly_avg, service_avg = average_payment_analysis(data)
        
        print("\nАнализ страховых услуг успешно завершен. Результаты сохранены в 'eda_results/unfiltered'.") 