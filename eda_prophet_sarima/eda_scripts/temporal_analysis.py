#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal analysis of insurance payment data without filtering
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
os.makedirs('eda_results/unfiltered/figures/temporal', exist_ok=True)

def load_unfiltered_data(filepath='data/unfiltered/processed_data.csv'):
    """
    Load the unfiltered processed dataset
    """
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
    """
    Analyze insurance payment trends over time (monthly)
    """
    print("\n=== ЕЖЕМЕСЯЧНЫЙ АНАЛИЗ СТРАХОВЫХ ВЫПЛАТ ===")
    
    # Extract year and month
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    df['month_name'] = df['service_date'].dt.strftime('%b')
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Aggregate by month
    monthly_payments = df.groupby('year_month').agg({
        'service_amount_net': ['sum', 'mean', 'count'],
        'service_document_id': 'nunique',
        'client_id': 'nunique'  # Changed from patient_id to client_id
    }).reset_index()
    
    # Flatten multi-level column names
    monthly_payments.columns = [
        'year_month' if col[0] == 'year_month' else 
        f"{col[0]}_{col[1]}" for col in monthly_payments.columns
    ]
    
    # Add date column for proper sorting
    monthly_payments['date'] = pd.to_datetime(monthly_payments['year_month'] + '-01')
    monthly_payments = monthly_payments.sort_values('date')
    
    # Calculate MoM (Month over Month) growth
    monthly_payments['payment_growth_mom_pct'] = monthly_payments['service_amount_net_sum'].pct_change() * 100
    
    # Calculate YoY (Year over Year) growth if we have multiple years
    years = df['year'].unique()
    if len(years) > 1:
        monthly_payments['month_num'] = pd.to_datetime(monthly_payments['date']).dt.month
        
        # Create a new DataFrame for YoY calculation
        yoy_data = []
        
        for month in range(1, 13):
            for year in sorted(years)[1:]:  # Skip the first year as it has no previous year
                current_month_data = monthly_payments[
                    (monthly_payments['month_num'] == month) & 
                    (pd.to_datetime(monthly_payments['date']).dt.year == year)
                ]
                
                prev_year_data = monthly_payments[
                    (monthly_payments['month_num'] == month) & 
                    (pd.to_datetime(monthly_payments['date']).dt.year == year - 1)
                ]
                
                if not current_month_data.empty and not prev_year_data.empty:
                    current_revenue = current_month_data['service_amount_net_sum'].values[0]
                    prev_revenue = prev_year_data['service_amount_net_sum'].values[0]
                    
                    yoy_growth = ((current_revenue - prev_revenue) / prev_revenue) * 100
                    
                    yoy_data.append({
                        'year': year,
                        'month': month,
                        'month_name': current_month_data['year_month'].values[0][5:],
                        'current_revenue': current_revenue,
                        'prev_revenue': prev_revenue,
                        'yoy_growth_pct': yoy_growth
                    })
        
        yoy_df = pd.DataFrame(yoy_data)
        
        # Save YoY data
        if not yoy_df.empty:
            yoy_df.to_csv('eda_results/unfiltered/data/yoy_growth.csv', index=False)
    
    # Save monthly payment data
    monthly_payments.to_csv('eda_results/unfiltered/data/monthly_payments.csv', index=False)
    
    # Plot monthly payment trends
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot as line chart with markers
    ax.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_sum'], 
           'o-', linewidth=3, markersize=8, 
           color=eda_style.COLORS['blue'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячный объем страховых выплат',
        xlabel='Месяц',
        ylabel='Сумма страховых выплат (руб.)'
    )
    
    # X-axis labels - use every n months for clarity
    n = 2 if len(monthly_payments) > 12 else 1
    ax.set_xticks(range(0, len(monthly_payments), n))
    ax.set_xticklabels([m for i, m in enumerate(monthly_payments['year_month']) if i % n == 0],
                     rotation=45, ha='right')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000000:.1f}M руб."))
    
    # Add annotations for key data points
    for i in range(0, len(monthly_payments), n):
        payment = monthly_payments['service_amount_net_sum'].iloc[i]
        ax.text(i, payment + (payment * 0.03), f"{payment/1000000:.1f}M руб.", 
               ha='center', fontsize=14, fontweight='bold')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/temporal/monthly_payments.png')
    
    # Plot monthly service count
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot as line chart with markers
    ax.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_count'], 
           'o-', linewidth=3, markersize=8, 
           color=eda_style.COLORS['green'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячное количество страховых выплат',
        xlabel='Месяц',
        ylabel='Количество выплат'
    )
    
    # X-axis labels
    ax.set_xticks(range(0, len(monthly_payments), n))
    ax.set_xticklabels([m for i, m in enumerate(monthly_payments['year_month']) if i % n == 0],
                     rotation=45, ha='right')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    
    # Add annotations for key data points
    for i in range(0, len(monthly_payments), n):
        count = monthly_payments['service_amount_net_count'].iloc[i]
        ax.text(i, count + (count * 0.03), f"{count:,.0f}", 
               ha='center', fontsize=14, fontweight='bold')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/temporal/monthly_service_count.png')
    
    # Plot monthly average payment
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot as line chart with markers
    ax.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_mean'], 
           'o-', linewidth=3, markersize=8, 
           color=eda_style.COLORS['cyan'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячная средняя сумма страховой выплаты',
        xlabel='Месяц',
        ylabel='Средняя выплата (руб.)'
    )
    
    # X-axis labels
    ax.set_xticks(range(0, len(monthly_payments), n))
    ax.set_xticklabels([m for i, m in enumerate(monthly_payments['year_month']) if i % n == 0],
                     rotation=45, ha='right')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Add annotations for key data points
    for i in range(0, len(monthly_payments), n):
        avg = monthly_payments['service_amount_net_mean'].iloc[i]
        ax.text(i, avg + (avg * 0.03), f"{avg:,.0f} руб.", 
               ha='center', fontsize=14, fontweight='bold')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/temporal/monthly_avg_payment.png')
    
    # NEW VISUALIZATION 1: Comparison of client numbers and mean aggregated payment over time
    fig, ax1 = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot clients count on the primary y-axis
    client_line = ax1.plot(range(len(monthly_payments)), monthly_payments['client_id_nunique'], 
             'o-', linewidth=3, markersize=8, 
             color=eda_style.COLORS['blue'], label='Количество клиентов')
    
    # Create a secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot mean payment on the secondary y-axis
    payment_line = ax2.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_mean'], 
             's-', linewidth=3, markersize=8, 
             color=eda_style.COLORS['green'], label='Средняя выплата')
    
    # Primary y-axis styling
    eda_style.apply_common_styles(
        ax1,
        title='Сравнение количества клиентов и средней суммы страховой выплаты',
        xlabel='Месяц',
        ylabel='Количество уникальных клиентов'
    )
    
    # Secondary y-axis styling
    ax2.set_ylabel('Средняя выплата (руб.)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(False)
    
    # Format axes
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # X-axis labels
    ax1.set_xticks(range(0, len(monthly_payments), n))
    ax1.set_xticklabels([m for i, m in enumerate(monthly_payments['year_month']) if i % n == 0],
                      rotation=45, ha='right')
    
    # Combine legends from both axes
    lines = client_line + payment_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=14, loc='upper left')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/temporal/clients_vs_avg_payment.png')
    
    # NEW VISUALIZATION 2: Comparison of payment numbers and mean payment over time
    fig, ax1 = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot payment count on the primary y-axis
    payment_count_line = ax1.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_count'], 
                                 'o-', linewidth=3, markersize=8, 
                                 color=eda_style.COLORS['blue'], label='Количество выплат')
    
    # Create a secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot mean payment on the secondary y-axis
    avg_payment_line = ax2.plot(range(len(monthly_payments)), monthly_payments['service_amount_net_mean'], 
                               's-', linewidth=3, markersize=8, 
                               color=eda_style.COLORS['green'], label='Средняя выплата')
    
    # Primary y-axis styling
    eda_style.apply_common_styles(
        ax1,
        title='Сравнение количества выплат и средней суммы страховой выплаты',
        xlabel='Месяц',
        ylabel='Количество выплат'
    )
    
    # Secondary y-axis styling
    ax2.set_ylabel('Средняя выплата (руб.)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(False)
    
    # Format axes
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # X-axis labels
    ax1.set_xticks(range(0, len(monthly_payments), n))
    ax1.set_xticklabels([m for i, m in enumerate(monthly_payments['year_month']) if i % n == 0],
                      rotation=45, ha='right')
    
    # Combine legends from both axes
    lines = payment_count_line + avg_payment_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=14, loc='upper left')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/temporal/payment_count_vs_avg_payment.png')
    
    return monthly_payments

def plot_year_over_year_comparison(df):
    """Plot year-over-year comparison for months with data in multiple years"""
    print("\n=== СРАВНЕНИЕ СТРАХОВЫХ ВЫПЛАТ ПО ГОДАМ ===")
    
    # Extract year and month for comparison
    df = df.copy()  # Create a copy to avoid modifying the original
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    df['month_name'] = df['service_date'].dt.strftime('%b')
    
    # Check if we have at least 2 years of data
    years = sorted(df['year'].unique())
    if len(years) < 2:
        print("Недостаточно данных для сравнения по годам. Нужны данные минимум за 2 года.")
        return
    
    # For each month, compare payments across available years
    yoy_results = []
    
    # Find months that have data in at least 2 years
    valid_months = []
    for month in range(1, 13):
        years_with_data = df[df['month'] == month]['year'].unique()
        if len(years_with_data) >= 2:
            valid_months.append(month)
    
    if not valid_months:
        print("Нет пересекающихся месяцев между годами для сравнения.")
        return
    
    # Aggregate data by year-month
    payments_by_month = df.groupby(['year', 'month']).agg({
        'service_amount_net': ['sum', 'mean', 'count'],
        'service_document_id': 'nunique',
        'client_id': 'nunique'  # Changed from patient_id to client_id
    }).reset_index()
    
    # Flatten multi-level column names
    payments_by_month.columns = [
        col[0] if col[0] in ['year', 'month'] else 
        f"{col[0]}_{col[1]}" for col in payments_by_month.columns
    ]
    
    # Add month name for better readability
    month_names = {
        1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн',
        7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
    }
    payments_by_month['month_name'] = payments_by_month['month'].map(month_names)
    
    # Save the year-month aggregated data
    payments_by_month.to_csv('eda_results/unfiltered/data/payments_by_year_month.csv', index=False)
    
    # Plot month-by-month comparison across years
    for month in valid_months:
        month_data = payments_by_month[payments_by_month['month'] == month].sort_values('year')
        
        if len(month_data) >= 2:  # Only plot if we have at least 2 years
            month_name = month_data['month_name'].iloc[0]
            
            # Plot total payments comparison
            fig, ax = eda_style.create_custom_figure(figsize=(12, 7))
            
            bars = ax.bar(month_data['year'].astype(str), month_data['service_amount_net_sum'],
                         color=[eda_style.COLORS['blue_grad_%d' % i] for i in range(len(month_data))])
            
            eda_style.apply_common_styles(
                ax,
                title=f'Сравнение объема страховых выплат в {month_name} по годам',
                xlabel='Год',
                ylabel='Сумма страховых выплат (руб.)'
            )
            
            # Format y-axis with currency
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000000:.1f}M руб."))
            
            # Add value labels on bars
            for i, v in enumerate(month_data['service_amount_net_sum']):
                ax.text(i, v + (v * 0.03), f"{v/1000000:.1f}M руб.", 
                       ha='center', fontsize=14, fontweight='bold')
            
            # Save the figure
            eda_style.save_figure(fig, f'eda_results/unfiltered/figures/temporal/yoy_payment_{month_name}.png')
            
            # If there are more than 2 years, calculate and display growth rates
            if len(month_data) > 2:
                # Create a figure for growth rates
                fig, ax = eda_style.create_custom_figure(figsize=(12, 7))
                
                # Calculate YoY growth rates
                prev_year_values = month_data['service_amount_net_sum'].iloc[:-1].values
                current_year_values = month_data['service_amount_net_sum'].iloc[1:].values
                growth_rates = ((current_year_values - prev_year_values) / prev_year_values) * 100
                
                years_pairs = [f"{y1}-{y2}" for y1, y2 in zip(month_data['year'].iloc[:-1], month_data['year'].iloc[1:])]
                
                # Define colors based on growth rate (positive/negative)
                bar_colors = [eda_style.COLORS['green'] if rate >= 0 else eda_style.COLORS['cyan'] for rate in growth_rates]
                
                bars = ax.bar(years_pairs, growth_rates, color=bar_colors)
                
                eda_style.apply_common_styles(
                    ax,
                    title=f'Темп роста страховых выплат в {month_name} год к году',
                    xlabel='Годы',
                    ylabel='Темп роста (%)'
                )
                
                # Add a reference line at 0%
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
                
                # Add value labels on bars
                for i, v in enumerate(growth_rates):
                    label_y = v + (5 if v >= 0 else -5)
                    va = 'bottom' if v >= 0 else 'top'
                    ax.text(i, label_y, f"{v:.1f}%", 
                           ha='center', va=va, fontsize=14, fontweight='bold')
                
                # Save the figure
                eda_style.save_figure(fig, f'eda_results/unfiltered/figures/temporal/yoy_growth_{month_name}.png')
    
    # Plot comparison of client count across years for each month
    for month in valid_months:
        month_data = payments_by_month[payments_by_month['month'] == month].sort_values('year')
        
        if len(month_data) >= 2:  # Only plot if we have at least 2 years
            month_name = month_data['month_name'].iloc[0]
            
            # Plot client count comparison
            fig, ax = eda_style.create_custom_figure(figsize=(12, 7))
            
            bars = ax.bar(month_data['year'].astype(str), month_data['client_id_nunique'],
                         color=[eda_style.COLORS['green_grad_%d' % min(i, 9)] for i in range(len(month_data))])
            
            eda_style.apply_common_styles(
                ax,
                title=f'Сравнение количества клиентов в {month_name} по годам',
                xlabel='Год',
                ylabel='Количество уникальных клиентов'
            )
            
            # Format y-axis with thousands separator
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
            
            # Add value labels on bars
            for i, v in enumerate(month_data['client_id_nunique']):
                ax.text(i, v + (v * 0.03), f"{v:,.0f}", 
                       ha='center', fontsize=14, fontweight='bold')
            
            # Save the figure
            eda_style.save_figure(fig, f'eda_results/unfiltered/figures/temporal/yoy_clients_{month_name}.png')
    
    return payments_by_month

if __name__ == "__main__":
    # Load unfiltered data
    data = load_unfiltered_data()
    
    if data is not None:
        # Analyze monthly payment trends
        monthly_data = monthly_payment_analysis(data)
        
        # Plot year-over-year comparison
        yoy_data = plot_year_over_year_comparison(data)
        
        print("\nАнализ временных трендов страховых выплат успешно завершен. Результаты сохранены в 'eda_results/unfiltered'.") 