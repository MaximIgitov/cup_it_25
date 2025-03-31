#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of client data and demographics for insurance services
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
os.makedirs('eda_results/unfiltered/figures/client', exist_ok=True)

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

def client_visit_analysis(df):
    """Analyze client visit patterns"""
    print("\n=== КЛИЕНТСКИЙ АНАЛИЗ ===")
    
    # Count visits per client
    visits_per_client = df.groupby('client_id')['service_document_id'].nunique().reset_index()
    visits_per_client.columns = ['client_id', 'visit_count']
    
    # Calculate summary statistics
    visit_stats = visits_per_client['visit_count'].describe()
    
    print("\nСтатистика посещений по клиентам:")
    print(f"Всего клиентов: {len(visits_per_client):,}")
    print(f"Среднее количество посещений на клиента: {visit_stats['mean']:.2f}")
    print(f"Медиана посещений на клиента: {visit_stats['50%']:.2f}")
    print(f"Максимальное количество посещений одним клиентом: {visit_stats['max']:.0f}")
    
    # Save client visit stats
    visits_per_client.to_csv('eda_results/unfiltered/data/client_visit_stats.csv', index=False)
    
    # Plot visit distribution
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Filter to reasonable range for better visualization
    plotting_data = visits_per_client[visits_per_client['visit_count'] <= 20]  # Focus on clients with ≤ 20 visits
    
    # Get visit counts
    visit_counts = plotting_data['visit_count'].value_counts().sort_index()
    
    # Plot as bar chart
    bars = ax.bar(visit_counts.index, visit_counts.values, color=sns.color_palette('viridis', len(visit_counts)))
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax, 
        title='Распределение клиентов по количеству посещений',
        xlabel='Количество посещений',
        ylabel='Количество клиентов'
    )
    
    # Add count labels on bars
    eda_style.value_label_bars(ax, fmt='{:,}', fontsize=14)
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Add x-ticks
    ax.set_xticks(np.arange(1, 21))
    
    # Save with consistent settings
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/client/client_visit_distribution.png')
    
    return visits_per_client

def client_retention_analysis(df):
    """Analyze client retention over time"""
    print("\n=== АНАЛИЗ УДЕРЖАНИЯ КЛИЕНТОВ ===")
    
    # Extract year-month and add to dataframe
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Get unique clients per month
    monthly_clients = df.groupby('year_month')['client_id'].nunique().reset_index()
    monthly_clients.columns = ['year_month', 'unique_clients']
    
    # Sort by year-month
    monthly_clients['date'] = pd.to_datetime(monthly_clients['year_month'] + '-01')
    monthly_clients = monthly_clients.sort_values('date')
    
    # Calculate new vs returning clients
    all_clients = set()
    new_clients = []
    returning_clients = []
    
    for _, row in monthly_clients.iterrows():
        ym = row['year_month']
        month_clients = set(df[df['year_month'] == ym]['client_id'].unique())
        
        # Calculate new clients
        new = month_clients - all_clients
        new_clients.append(len(new))
        
        # Calculate returning clients
        returning = month_clients & all_clients
        returning_clients.append(len(returning))
        
        # Update all_clients
        all_clients.update(month_clients)
    
    # Add calculated values to dataframe
    monthly_clients['new_clients'] = new_clients
    monthly_clients['returning_clients'] = returning_clients
    
    # Save retention data
    monthly_clients.to_csv('eda_results/unfiltered/data/client_retention.csv', index=False)
    
    # Plot new vs returning clients
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Create stacked bar chart
    ax.bar(range(len(monthly_clients)), monthly_clients['new_clients'], label='Новые клиенты',
           color=eda_style.COLORS['blue'])
    ax.bar(range(len(monthly_clients)), monthly_clients['returning_clients'], 
           bottom=monthly_clients['new_clients'], label='Повторные клиенты',
           color=eda_style.COLORS['green'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Удержание клиентов: новые и повторные клиенты',
        xlabel='Месяц',
        ylabel='Количество клиентов'
    )
    
    # X-axis labels (use every other month for clarity)
    step = 2 if len(monthly_clients) > 12 else 1
    ax.set_xticks(range(0, len(monthly_clients), step))
    ax.set_xticklabels([m for i, m in enumerate(monthly_clients['year_month']) if i % step == 0],
                     rotation=45, ha='right')
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Add legend with better positioning
    ax.legend(fontsize=14, loc='upper left')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/client/client_retention.png')
    
    # Calculate and plot retention rate
    monthly_clients['retention_rate'] = monthly_clients['returning_clients'] / monthly_clients['unique_clients'] * 100
    
    # Create the retention rate figure
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    ax.plot(range(len(monthly_clients)), monthly_clients['retention_rate'], 
            marker='o', linestyle='-', linewidth=3, color=eda_style.COLORS['cyan'])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячный показатель удержания клиентов',
        xlabel='Месяц',
        ylabel='Коэффициент удержания (%)'
    )
    
    # X-axis labels
    ax.set_xticks(range(0, len(monthly_clients), step))
    ax.set_xticklabels([m for i, m in enumerate(monthly_clients['year_month']) if i % step == 0],
                     rotation=45, ha='right')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
    
    # Add percentage labels
    for i, rate in enumerate(monthly_clients['retention_rate']):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontweight='bold', fontsize=14)
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/client/retention_rate.png')
    
    return monthly_clients

def service_mix_analysis(df):
    """Analyze service type mix by client segment"""
    print("\n=== АНАЛИЗ СТРУКТУРЫ УСЛУГ ===")
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Note: 'service_type_id' column not found, using 'service_code' as service type indicator")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Group clients by visit frequency
    visits_per_client = df.groupby('client_id')['service_document_id'].nunique().reset_index()
    visits_per_client.columns = ['client_id', 'visit_count']
    
    # Define client segments
    visits_per_client['segment'] = pd.cut(
        visits_per_client['visit_count'], 
        bins=[0, 1, 3, 10, float('inf')],
        labels=['Разовые', 'Случайные', 'Регулярные', 'Активные']
    )
    
    # Merge segment info back to df
    df_with_segments = pd.merge(df, visits_per_client[['client_id', 'segment']], 
                               left_on='client_id', right_on='client_id')
    
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
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot the data with custom colors
    colors = [eda_style.COLORS['blue'], eda_style.COLORS['cyan'], 
              eda_style.COLORS['green'], eda_style.COLORS['lime'], 
              eda_style.COLORS['yellow']]
    
    service_mix_plot.plot(kind='bar', stacked=False, ax=ax, color=colors[:len(top_services)])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Структура услуг по сегментам клиентов',
        xlabel='Сегмент клиентов',
        ylabel='Использование услуг (%)'
    )
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_percent))
    
    # Improve legend
    ax.legend(title='Тип услуги', fontsize=14, title_fontsize=16, loc='upper right')
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/client/segment_service_mix.png')
    
    # Calculate average payment per segment
    segment_payment = df_with_segments.groupby('segment')['service_amount_net'].agg(['sum', 'mean']).reset_index()
    
    # Plot average payment per segment
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Create bar chart with custom colors
    bars = ax.bar(segment_payment['segment'], segment_payment['mean'],
                 color=[eda_style.COLORS['blue_dark'], eda_style.COLORS['blue'], 
                        eda_style.COLORS['cyan'], eda_style.COLORS['green']])
    
    # Apply common styles
    eda_style.apply_common_styles(
        ax,
        title='Средняя выплата на визит по сегментам клиентов',
        xlabel='Сегмент клиентов',
        ylabel='Средняя выплата (руб.)'
    )
    
    # Format y-axis with currency
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Add value labels
    eda_style.value_label_bars(ax, fmt='{:,.0f} руб.', fontsize=14)
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/client/segment_payment.png')
    
    return segment_service_mix

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data()
    
    if data is not None:
        # Perform client visit analysis
        visit_data = client_visit_analysis(data)
        
        # Perform retention analysis
        retention_data = client_retention_analysis(data)
        
        # Perform service mix analysis
        service_mix_data = service_mix_analysis(data)
        
        print("\nАнализ клиентов успешно завершен. Результаты сохранены в 'eda_results/unfiltered'.") 