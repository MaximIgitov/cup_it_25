#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time series analysis of insurance payments with monthly trends
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
os.makedirs('eda_results/unfiltered/figures/time_series', exist_ok=True)

def load_processed_data(filepath='data/unfiltered/processed_data.csv'):
    """Load the processed dataset"""
    print(f"Загрузка обработанных данных из {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['service_date'] = pd.to_datetime(df['service_date'])
        # Ensure service_amount_net is numeric
        df['service_amount_net'] = pd.to_numeric(df['service_amount_net'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Файл {filepath} не найден.")
        print("Пожалуйста, сначала запустите eda_scripts/eda_main.py.")
        return None

def services_time_series(df):
    """Create time series of service counts and average payments by month"""
    print("\n=== ВРЕМЕННОЙ РЯД УСЛУГ И СРЕДНИХ ВЫПЛАТ ===")
    
    # Extract year-month and add to dataframe
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Check if we have service_type_id column, if not use service_code instead
    if 'service_type_id' not in df.columns:
        print("Примечание: колонка 'service_type_id' не найдена, используется 'service_code' как индикатор типа услуги")
        service_type_column = 'service_code'
    else:
        service_type_column = 'service_type_id'
    
    # Group by month
    monthly_data = df.groupby('year_month').agg({
        'service_amount_net': ['mean', 'sum'],
        service_type_column: 'nunique',
        'service_document_id': 'count'
    }).reset_index()
    
    # Flatten multi-level column names
    monthly_data.columns = [
        'year_month' if col[0] == 'year_month' else 
        f"{col[0]}_{col[1]}" for col in monthly_data.columns
    ]
    
    # Calculate average payment per service (this ensures payment * count = sum)
    monthly_data['avg_payment_per_service'] = monthly_data['service_amount_net_sum'] / monthly_data['service_document_id_count']
    
    # Add date column for proper sorting
    monthly_data['date'] = pd.to_datetime(monthly_data['year_month'] + '-01')
    
    # Explicit sorting by date to ensure correct year order
    monthly_data = monthly_data.sort_values('date')
    
    # Verify correct year order
    years = monthly_data['year_month'].str[:4].unique()
    print(f"Годы в данных (в порядке сортировки): {', '.join(years)}")
    
    # Save aggregated data
    monthly_data.to_csv('eda_results/unfiltered/data/services_time_series.csv', index=False)
    
    # Create figure
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot total payment on primary y-axis
    line1 = ax1.plot(
        range(len(monthly_data)), 
        monthly_data['service_amount_net_sum'],
        marker='o', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Суммарная выплата'
    )
    
    # Format y-axis with currency
    ax1.set_ylabel('Суммарная выплата (руб.)', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)

    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot service count on secondary y-axis
    line2 = ax2.plot(
        range(len(monthly_data)), 
        monthly_data['service_document_id_count'],
        marker='s', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество услуг'
    )
    
    # Create tertiary y-axis for average payment per service
    ax3 = ax1.twinx()
    # Offset the axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot average payment per service on tertiary y-axis
    line3 = ax3.plot(
        range(len(monthly_data)), 
        monthly_data['avg_payment_per_service'],
        marker='d', 
        markersize=8,
        linewidth=3,
        linestyle='--',
        color=eda_style.COLORS['yellow'],
        label='Средняя выплата на услугу'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество услуг', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Format tertiary y-axis
    ax3.set_ylabel('Средняя выплата на услугу (руб.)', 
                   fontsize=16, fontweight='bold', color=eda_style.COLORS['yellow'])
    ax3.tick_params(axis='y', labelcolor=eda_style.COLORS['yellow'], labelsize=14)
    ax3.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # X-axis formatting
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества услуг и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # X-ticks as months
    step = 2 if len(monthly_data) > 18 else 1
    ax1.set_xticks(range(0, len(monthly_data), step))
    ax1.set_xticklabels(
        [m for i, m in enumerate(monthly_data['year_month']) if i % step == 0],
        rotation=45, 
        ha='right',
        fontsize=16
    )
    
    # Add year separators and labels
    if len(years) > 1:
        for i, year in enumerate(years[1:], 1):
            # Find first month of this year
            year_start_idx = monthly_data[monthly_data['year_month'].str.startswith(year)].index.min()
            if year_start_idx is not None:
                idx = monthly_data.index.get_loc(year_start_idx)
                ax1.axvline(x=idx-0.5, color='red', linestyle='-', linewidth=2, alpha=0.5)
                ax1.text(idx-0.5, ax1.get_ylim()[1]*0.98, year, fontsize=16, 
                         ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=16, frameon=True, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/time_series/services_time_series.png')
    
    # Verify the calculation: average * count = total
    product = monthly_data['avg_payment_per_service'] * monthly_data['service_document_id_count']
    difference = product - monthly_data['service_amount_net_sum']
    max_diff_pct = (abs(difference) / monthly_data['service_amount_net_sum'] * 100).max()
    print(f"Проверка расчета: макс. разница между произведением и суммой: {max_diff_pct:.6f}%")
    
    return monthly_data

def clients_time_series(df):
    """Create time series of client counts and average payments by month"""
    print("\n=== ВРЕМЕННОЙ РЯД КЛИЕНТОВ И СРЕДНИХ ВЫПЛАТ ===")
    
    # Extract year-month and add to dataframe
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Group by month
    monthly_data = df.groupby('year_month').agg({
        'service_amount_net': ['mean', 'sum'],
        'client_id': 'nunique',
        'service_document_id': 'count'  # Add count of services
    }).reset_index()
    
    # Flatten multi-level column names
    monthly_data.columns = [
        'year_month' if col[0] == 'year_month' else 
        f"{col[0]}_{col[1]}" for col in monthly_data.columns
    ]
    
    # Calculate average payment per client
    monthly_data['avg_payment_per_client'] = monthly_data['service_amount_net_sum'] / monthly_data['client_id_nunique']
    
    # Calculate average payment per service
    monthly_data['avg_payment_per_service'] = monthly_data['service_amount_net_sum'] / monthly_data['service_document_id_count']
    
    # Add date column for proper sorting
    monthly_data['date'] = pd.to_datetime(monthly_data['year_month'] + '-01')
    
    # Explicit sorting by date to ensure correct year order
    monthly_data = monthly_data.sort_values('date')
    
    # Verify correct year order
    years = monthly_data['year_month'].str[:4].unique()
    print(f"Годы в данных (в порядке сортировки): {', '.join(years)}")
    
    # Save aggregated data
    monthly_data.to_csv('eda_results/unfiltered/data/clients_time_series.csv', index=False)
    
    # Create dual-axis figure
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot average payment per client on primary y-axis
    line1 = ax1.plot(
        range(len(monthly_data)), 
        monthly_data['avg_payment_per_client'],
        marker='o', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Средняя выплата на клиента'
    )
    
    # Format y-axis with currency
    ax1.set_ylabel('Средняя выплата на клиента (руб.)', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)

    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot client count on secondary y-axis
    line2 = ax2.plot(
        range(len(monthly_data)), 
        monthly_data['client_id_nunique'],
        marker='s', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество уникальных клиентов'
    )
    
    # Create tertiary y-axis for total sum
    ax3 = ax1.twinx()
    # Offset the axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot sum on tertiary y-axis
    line3 = ax3.plot(
        range(len(monthly_data)), 
        monthly_data['service_amount_net_sum'] / 1e6,  # Scale for readability
        marker='d', 
        markersize=8,
        linewidth=3,
        linestyle='--',
        color=eda_style.COLORS['yellow'],
        label='Суммарная выплата (млн руб.)'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество клиентов', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Format tertiary y-axis
    ax3.set_ylabel('Суммарная выплата (млн руб.)', 
                   fontsize=16, fontweight='bold', color=eda_style.COLORS['yellow'])
    ax3.tick_params(axis='y', labelcolor=eda_style.COLORS['yellow'], labelsize=14)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    
    # X-axis formatting
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества клиентов и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # X-ticks as months
    step = 2 if len(monthly_data) > 18 else 1
    ax1.set_xticks(range(0, len(monthly_data), step))
    ax1.set_xticklabels(
        [m for i, m in enumerate(monthly_data['year_month']) if i % step == 0],
        rotation=45, 
        ha='right',
        fontsize=16
    )
    
    # Add year separators and labels
    if len(years) > 1:
        for i, year in enumerate(years[1:], 1):
            # Find first month of this year
            year_start_idx = monthly_data[monthly_data['year_month'].str.startswith(year)].index.min()
            if year_start_idx is not None:
                idx = monthly_data.index.get_loc(year_start_idx)
                ax1.axvline(x=idx-0.5, color='red', linestyle='-', linewidth=2, alpha=0.5)
                ax1.text(idx-0.5, ax1.get_ylim()[1]*0.98, year, fontsize=16, 
                         ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=16, frameon=True, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/time_series/clients_time_series.png')
    
    # Verify the calculation: avg_per_client * client_count ≈ total
    product = monthly_data['avg_payment_per_client'] * monthly_data['client_id_nunique']
    difference = product - monthly_data['service_amount_net_sum']
    max_diff_pct = (abs(difference) / monthly_data['service_amount_net_sum'] * 100).max()
    print(f"Проверка расчета: макс. разница между произведением и суммой: {max_diff_pct:.6f}%")
    
    return monthly_data

def documents_time_series(df):
    """Create time series of document counts and average payments by month"""
    print("\n=== ВРЕМЕННОЙ РЯД ДОКУМЕНТОВ И СРЕДНИХ ВЫПЛАТ ===")
    
    # Extract year-month and add to dataframe
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    
    # Group by month
    monthly_data = df.groupby('year_month').agg({
        'service_amount_net': ['mean', 'sum'],
        'service_document_id': ['nunique', 'count']  # Both unique count and total count
    }).reset_index()
    
    # Flatten multi-level column names
    monthly_data.columns = [
        'year_month' if col[0] == 'year_month' else 
        f"{col[0]}_{col[1]}" for col in monthly_data.columns
    ]
    
    # Calculate average payment per document
    monthly_data['avg_payment_per_document'] = monthly_data['service_amount_net_sum'] / monthly_data['service_document_id_nunique']
    
    # Calculate average payment per service
    monthly_data['avg_payment_per_service'] = monthly_data['service_amount_net_sum'] / monthly_data['service_document_id_count']
    
    # Add date column for proper sorting
    monthly_data['date'] = pd.to_datetime(monthly_data['year_month'] + '-01')
    
    # Explicit sorting by date to ensure correct year order
    monthly_data = monthly_data.sort_values('date')
    
    # Verify correct year order
    years = monthly_data['year_month'].str[:4].unique()
    print(f"Годы в данных (в порядке сортировки): {', '.join(years)}")
    
    # Save aggregated data
    monthly_data.to_csv('eda_results/unfiltered/data/documents_time_series.csv', index=False)
    
    # Create dual-axis figure
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot average payment per document on primary y-axis
    line1 = ax1.plot(
        range(len(monthly_data)), 
        monthly_data['avg_payment_per_document'],
        marker='o', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Средняя выплата на документ'
    )
    
    # Format y-axis with currency
    ax1.set_ylabel('Средняя выплата на документ (руб.)', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)

    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot document count on secondary y-axis
    line2 = ax2.plot(
        range(len(monthly_data)), 
        monthly_data['service_document_id_nunique'],
        marker='s', 
        markersize=10,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество документов'
    )
    
    # Create tertiary y-axis for total sum
    ax3 = ax1.twinx()
    # Offset the axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot total sum on tertiary y-axis
    line3 = ax3.plot(
        range(len(monthly_data)), 
        monthly_data['service_amount_net_sum'] / 1e6,  # Scale for readability
        marker='d', 
        markersize=8,
        linewidth=3,
        linestyle='--',
        color=eda_style.COLORS['yellow'],
        label='Суммарная выплата (млн руб.)'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество документов', 
                  fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Format tertiary y-axis
    ax3.set_ylabel('Суммарная выплата (млн руб.)', 
                   fontsize=16, fontweight='bold', color=eda_style.COLORS['yellow'])
    ax3.tick_params(axis='y', labelcolor=eda_style.COLORS['yellow'], labelsize=14)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    
    # X-axis formatting
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества документов и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # X-ticks as months
    step = 2 if len(monthly_data) > 18 else 1
    ax1.set_xticks(range(0, len(monthly_data), step))
    ax1.set_xticklabels(
        [m for i, m in enumerate(monthly_data['year_month']) if i % step == 0],
        rotation=45, 
        ha='right',
        fontsize=16
    )
    
    # Add year separators and labels
    if len(years) > 1:
        for i, year in enumerate(years[1:], 1):
            # Find first month of this year
            year_start_idx = monthly_data[monthly_data['year_month'].str.startswith(year)].index.min()
            if year_start_idx is not None:
                idx = monthly_data.index.get_loc(year_start_idx)
                ax1.axvline(x=idx-0.5, color='red', linestyle='-', linewidth=2, alpha=0.5)
                ax1.text(idx-0.5, ax1.get_ylim()[1]*0.98, year, fontsize=16, 
                         ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=16, frameon=True, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/time_series/documents_time_series.png')
    
    # Verify the calculation: avg_per_document * document_count ≈ total
    product = monthly_data['avg_payment_per_document'] * monthly_data['service_document_id_nunique']
    difference = product - monthly_data['service_amount_net_sum']
    max_diff_pct = (abs(difference) / monthly_data['service_amount_net_sum'] * 100).max()
    print(f"Проверка расчета: макс. разница между произведением и суммой: {max_diff_pct:.6f}%")
    
    return monthly_data

def service_code_analysis(df):
    """Analyze and compare service codes by count and payment sum"""
    print("\n=== АНАЛИЗ КОДОВ УСЛУГ И СУММ ВЫПЛАТ ===")
    
    # Group by service_code
    service_data = df.groupby('service_code').agg({
        'service_amount_net': 'sum',
        'service_document_id': 'count'
    }).reset_index()
    
    # Sort by payment sum descending
    service_data_by_payment = service_data.sort_values('service_amount_net', ascending=False)
    service_data_by_count = service_data.sort_values('service_document_id', ascending=False)
    
    # Save aggregated data
    service_data_by_payment.to_csv('eda_results/unfiltered/data/service_code_payments.csv', index=False)
    service_data_by_count.to_csv('eda_results/unfiltered/data/service_code_counts.csv', index=False)
    
    # Create figure for top services by payment
    fig1, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Create bar plot for top services by payment
    top_n = 20  # Top services to show
    plot_data = service_data_by_payment.head(top_n)
    
    # Plot
    bars = ax1.bar(
        range(len(plot_data)),
        plot_data['service_amount_net'],
        color=eda_style.COLORS['blue'],
        alpha=0.8
    )
    
    # Add service count labels on top of bars
    for i, bar in enumerate(bars):
        count = plot_data['service_document_id'].iloc[i]
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05 * max(plot_data['service_amount_net']),
            f'Кол-во: {count:,}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            rotation=0
        )
    
    # Format axes
    ax1.set_xlabel('Код услуги', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Сумма выплат (руб.)', fontsize=20, fontweight='bold')
    ax1.set_title('Топ-20 кодов услуг по суммарным выплатам', fontsize=24, fontweight='bold', pad=20)
    
    # X-ticks formatting
    ax1.set_xticks(range(len(plot_data)))
    ax1.set_xticklabels(
        plot_data['service_code'],
        rotation=45,
        ha='right',
        fontsize=14
    )
    
    # Y-axis currency formatting
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Grid for readability
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    eda_style.save_figure(fig1, 'eda_results/unfiltered/figures/service_code_payments.png')
    
    # Create figure for top services by count
    fig2, ax2 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Create bar plot for top services by count
    plot_data = service_data_by_count.head(top_n)
    
    # Plot
    bars = ax2.bar(
        range(len(plot_data)),
        plot_data['service_document_id'],
        color=eda_style.COLORS['green'],
        alpha=0.8
    )
    
    # Add payment labels on top of bars
    for i, bar in enumerate(bars):
        payment = plot_data['service_amount_net'].iloc[i]
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05 * max(plot_data['service_document_id']),
            f'{eda_style.format_currency(payment, None)} руб.',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            rotation=0
        )
    
    # Format axes
    ax2.set_xlabel('Код услуги', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Количество услуг', fontsize=20, fontweight='bold')
    ax2.set_title('Топ-20 кодов услуг по количеству', fontsize=24, fontweight='bold', pad=20)
    
    # X-ticks formatting
    ax2.set_xticks(range(len(plot_data)))
    ax2.set_xticklabels(
        plot_data['service_code'],
        rotation=45,
        ha='right',
        fontsize=14
    )
    
    # Y-axis count formatting
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # Grid for readability
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    eda_style.save_figure(fig2, 'eda_results/unfiltered/figures/service_code_counts.png')
    
    # Create pie chart for top 10 services by payment
    fig3, ax3 = eda_style.create_custom_figure(figsize=(14, 14))
    
    # Get top 10 services and group others
    top_10 = service_data_by_payment.head(10).copy()
    others_sum = service_data_by_payment.iloc[10:]['service_amount_net'].sum()
    others_count = service_data_by_payment.iloc[10:]['service_document_id'].sum()
    
    # Add "Others" category
    top_10.loc[len(top_10)] = ['Другие', others_sum, others_count]
    
    # Use available colors from the COLORS dictionary
    pie_colors = [
        eda_style.COLORS['blue'], 
        eda_style.COLORS['cyan'],
        eda_style.COLORS['green'], 
        eda_style.COLORS['lime'],
        eda_style.COLORS['yellow'], 
        eda_style.COLORS['orange'],
        eda_style.COLORS['red'], 
        eda_style.COLORS['blue_dark'],
        eda_style.COLORS['blue_grad_3'],
        eda_style.COLORS['green_grad_3'],
        eda_style.COLORS['blue_grad_7']
    ]
    
    # Plot pie chart
    wedges, texts, autotexts = ax3.pie(
        top_10['service_amount_net'], 
        labels=None,
        autopct='',
        startangle=90,
        colors=pie_colors
    )
    
    # Add legend with percentages and counts
    total_payment = top_10['service_amount_net'].sum()
    labels = [f"{code}: {eda_style.format_currency(payment, None)} руб. ({payment/total_payment:.1%}, {count:,} услуг)" 
              for code, payment, count in zip(top_10['service_code'], top_10['service_amount_net'], top_10['service_document_id'])]
    
    ax3.legend(wedges, labels, title="Коды услуг", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14)
    
    # Set title
    ax3.set_title('Распределение выплат по кодам услуг (Топ-10)', fontsize=24, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    eda_style.save_figure(fig3, 'eda_results/unfiltered/figures/service_code_pie.png')
    
    print(f"Проанализировано {len(service_data)} уникальных кодов услуг")
    print(f"Топ-5 кодов услуг по суммам выплат:")
    for _, row in service_data_by_payment.head(5).iterrows():
        print(f"  Код {row['service_code']}: {eda_style.format_currency(row['service_amount_net'], None)} руб. ({row['service_document_id']} услуг)")
    
    print(f"\nТоп-5 кодов услуг по количеству:")
    for _, row in service_data_by_count.head(5).iterrows():
        print(f"  Код {row['service_code']}: {row['service_document_id']} услуг ({eda_style.format_currency(row['service_amount_net'], None)} руб.)")
    
    return service_data_by_payment, service_data_by_count

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data()
    
    if data is not None:
        # Generate time series for services
        services_data = services_time_series(data)
        
        # Generate time series for clients
        clients_data = clients_time_series(data)
        
        # Generate time series for documents
        documents_data = documents_time_series(data)
        
        # Analyze service codes
        service_code_data = service_code_analysis(data)
        
        print("\nАнализ временных рядов и кодов услуг успешно завершен. Результаты сохранены в 'eda_results/unfiltered'.") 