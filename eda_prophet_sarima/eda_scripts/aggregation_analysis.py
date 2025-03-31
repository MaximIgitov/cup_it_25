#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Агрегационный анализ страховых выплат
Правильная агрегация по месяцам и service_code
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter

# Import custom styling
import eda_style

# Create necessary directories
os.makedirs('eda_results/unfiltered/data', exist_ok=True)
os.makedirs('eda_results/unfiltered/figures/aggregation', exist_ok=True)

def load_raw_data(filepath='CupIT_Sber_data.csv'):
    """Load the raw dataset directly with handling for different delimiters and encodings"""
    print(f"Загрузка исходных данных из {filepath}...")
    
    # Try different delimiters and encodings
    delimiters = [';', ',']
    encodings = ['cp1251', 'utf-8']
    
    for delimiter in delimiters:
        for encoding in encodings:
            try:
                print(f"Попытка загрузки с разделителем '{delimiter}' и кодировкой '{encoding}'...")
                df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                print(f"Успешно загружено с разделителем '{delimiter}' и кодировкой '{encoding}'")
                
                # Convert service_date to datetime
                df['service_date'] = pd.to_datetime(df['service_date'])
                # Ensure amount is numeric
                df['service_amount_net'] = pd.to_numeric(df['service_amount_net'].str.replace(',', '.'), errors='coerce')                
                # Ensure service_code is string
                df['service_code'] = df['service_code'].astype(str)
                
                # Basic info about the dataset
                print(f"Загружено {len(df)} строк данных")
                print(f"Временной диапазон: {df['service_date'].min().strftime('%Y-%m-%d')} - {df['service_date'].max().strftime('%Y-%m-%d')}")
                
                # Rename patient_id to client_id if needed
                if 'patient_id' in df.columns and 'client_id' not in df.columns:
                    df['client_id'] = df['patient_id']
                
                return df
            except Exception as e:
                print(f"Не удалось загрузить с разделителем '{delimiter}' и кодировкой '{encoding}': {e}")
    

def monthly_aggregations(df):
    """Perform monthly aggregations of the data"""
    print("\n=== MONTHLY AGGREGATION ===")
    
    # Verify datetime conversion and show sample data
    df['service_date'] = pd.to_datetime(df['service_date'], format='%Y-%m-%d')
    print("Sample dates (post-conversion):")
    print(df['service_date'].head())

    # Create year-month grouping column
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    df['month_name'] = df['service_date'].dt.strftime('%b')
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')

    # Group by year_month and aggregate
    monthly_data = df.groupby('year_month').agg({
        'service_amount_net': ['sum', 'mean', 'count'],
        'service_code': 'nunique',
        'service_document_id': 'nunique',
        'client_id': 'nunique'
    }).reset_index()

    # Flatten multi-level columns
    monthly_data.columns = ['_'.join(col).strip('_') for col in monthly_data.columns.values]
    
    # Calculate additional metrics
    monthly_data['avg_payment_per_service'] = (
        monthly_data['service_amount_net_sum'] / 
        monthly_data['service_amount_net_count']
    )
    monthly_data['avg_payment_per_service_type'] = (
        monthly_data['service_amount_net_sum'] / 
        monthly_data['service_code_nunique']
    )
    monthly_data['avg_payment_per_document'] = (
        monthly_data['service_amount_net_sum'] / 
        monthly_data['service_document_id_nunique']
    )
    monthly_data['avg_payment_per_client'] = (
        monthly_data['service_amount_net_sum'] / 
        monthly_data['client_id_nunique']
    )

    monthly_data['date'] = pd.to_datetime(monthly_data['year_month'] + '-01')
    monthly_data = monthly_data.sort_values('date')

    monthly_data['year'] = monthly_data['date'].dt.year
    monthly_data['month'] = monthly_data['date'].dt.month

    # Save results
    monthly_data.to_csv('monthly_aggregation.csv', index=False)
    print(f"Results saved to monthly_aggregation.csv")

    return monthly_data


def service_code_aggregations(df):
    """Perform aggregations by service_code"""
    print("\n=== АГРЕГАЦИЯ ПО КОДАМ УСЛУГ ===")
    
    # Group by service_code
    service_data = df.groupby('service_code').agg({
        'service_amount_net': ['sum', 'mean', 'count'],
        'service_document_id': 'nunique',
        'client_id': 'nunique'
    }).reset_index()
    
    # Flatten multi-level column names
    service_data.columns = [
        'service_code' if col[0] == 'service_code' else 
        f"{col[0]}_{col[1]}" for col in service_data.columns
    ]
    
    # Sort by payment sum and count
    service_data_by_sum = service_data.sort_values('service_amount_net_sum', ascending=False)
    service_data_by_count = service_data.sort_values('service_amount_net_count', ascending=False)
    
    # Save aggregated data
    service_data_by_sum.to_csv('eda_results/unfiltered/data/service_code_by_payment.csv', index=False)
    service_data_by_count.to_csv('eda_results/unfiltered/data/service_code_by_count.csv', index=False)
    
    print(f"Найдено {len(service_data)} уникальных кодов услуг")
    print(f"Агрегации по кодам услуг сохранены в:")
    print(f"  - eda_results/unfiltered/data/service_code_by_payment.csv")
    print(f"  - eda_results/unfiltered/data/service_code_by_count.csv")
    
    return service_data_by_sum, service_data_by_count

def plot_monthly_service_metrics(monthly_data):
    """Plot monthly service metrics with clear year separation"""
    print("\n=== ВИЗУАЛИЗАЦИЯ ПОКАЗАТЕЛЕЙ УСЛУГ ПО МЕСЯЦАМ ===")
    
    # Create figure with proper date x-axis
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot total payment amount
    line1 = ax1.plot(
        monthly_data['date'], 
        monthly_data['avg_payment_per_service_type'],
        marker='o', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Суммарная выплата'
    )
    
    # Format primary y-axis with currency
    ax1.set_ylabel('Суммарная выплата (руб.)', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot service count
    line2 = ax2.plot(
        monthly_data['date'], 
        monthly_data['service_code_nunique'],
        marker='s', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество услуг'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество услуг', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    
    # X-axis formatting with proper date display
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества услуг и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # Format x-axis with month-year labels
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45, labelsize=14)
    
    # Add year separators
    years = sorted(monthly_data['year'].unique())
    
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=16, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/aggregation/services_monthly_trend.png')
    
    return fig

def plot_monthly_client_metrics(monthly_data):
    """Plot monthly client metrics with clear year separation"""
    print("\n=== ВИЗУАЛИЗАЦИЯ ПОКАЗАТЕЛЕЙ КЛИЕНТОВ ПО МЕСЯЦАМ ===")
    
    # Create figure with proper date x-axis
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot average payment per client
    line1 = ax1.plot(
        monthly_data['date'], 
        monthly_data['avg_payment_per_client'],
        marker='o', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Средняя выплата на клиента'
    )
    
    # Format primary y-axis with currency
    ax1.set_ylabel('Средняя выплата на клиента (руб.)', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot client count
    line2 = ax2.plot(
        monthly_data['date'], 
        monthly_data['client_id_nunique'],
        marker='s', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество уникальных клиентов'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество клиентов', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    
    # X-axis formatting with proper date display
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества клиентов и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # Format x-axis with month-year labels
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45, labelsize=14)
    
    # Add year separators
    years = sorted(monthly_data['year'].unique())
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=16, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/aggregation/clients_monthly_trend.png')
    
    return fig

def plot_monthly_document_metrics(monthly_data):
    """Plot monthly document metrics with clear year separation"""
    print("\n=== ВИЗУАЛИЗАЦИЯ ПОКАЗАТЕЛЕЙ ДОКУМЕНТОВ ПО МЕСЯЦАМ ===")
    
    # Create figure with proper date x-axis
    fig, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Plot average payment per document
    line1 = ax1.plot(
        monthly_data['date'], 
        monthly_data['avg_payment_per_document'],
        marker='o', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['blue'],
        label='Средняя выплата на документ'
    )
    
    # Format primary y-axis with currency
    ax1.set_ylabel('Средняя выплата на документ (руб.)', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['blue'])
    ax1.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    ax1.tick_params(axis='y', labelcolor=eda_style.COLORS['blue'], labelsize=16)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot document count
    line2 = ax2.plot(
        monthly_data['date'], 
        monthly_data['service_document_id_nunique'],
        marker='s', 
        markersize=8,
        linewidth=3,
        color=eda_style.COLORS['green'],
        label='Количество документов'
    )
    
    # Format secondary y-axis
    ax2.set_ylabel('Количество документов', 
                   fontsize=20, fontweight='bold', color=eda_style.COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=eda_style.COLORS['green'], labelsize=16)
    ax2.yaxis.set_major_formatter(FuncFormatter(eda_style.format_thousands))
    
    # X-axis formatting with proper date display
    ax1.set_xlabel('Месяц', fontsize=20, fontweight='bold')
    ax1.set_title('Динамика количества документов и выплат по месяцам', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # Format x-axis with month-year labels
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45, labelsize=14)
    
    # Add year separators
    years = sorted(monthly_data['year'].unique())
    
    
    # Grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=16, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/aggregation/documents_monthly_trend.png')
    
    return fig

def plot_top_service_codes(service_data_by_sum, service_data_by_count):
    """Plot top service codes by payment and count"""
    print("\n=== ВИЗУАЛИЗАЦИЯ ТОП КОДОВ УСЛУГ ===")
    
    # Top N service codes to show
    top_n = 20
    
    # 1. Plot top services by payment sum
    fig1, ax1 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Get top N data
    plot_data = service_data_by_sum.head(top_n).copy()
    
    # Plot
    bars = ax1.bar(
        range(len(plot_data)),
        plot_data['service_amount_net_sum'],
        color=eda_style.COLORS['blue'],
        alpha=0.8
    )
    
    # Add service count labels
    for i, bar in enumerate(bars):
        count = plot_data['service_amount_net_count'].iloc[i]
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05 * plot_data['service_amount_net_sum'].max(),
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
    ax1.set_title(f'Топ-{top_n} кодов услуг по суммарным выплатам', fontsize=24, fontweight='bold', pad=20)
    
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
    eda_style.save_figure(fig1, 'eda_results/unfiltered/figures/aggregation/top_services_by_payment.png')
    
    # 2. Plot top services by count
    fig2, ax2 = eda_style.create_custom_figure(figsize=(16, 10))
    
    # Get top N data
    plot_data = service_data_by_count.head(top_n).copy()
    
    # Plot
    bars = ax2.bar(
        range(len(plot_data)),
        plot_data['service_amount_net_count'],
        color=eda_style.COLORS['green'],
        alpha=0.8
    )
    
    # Add payment sum labels
    for i, bar in enumerate(bars):
        payment = plot_data['service_amount_net_sum'].iloc[i]
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05 * plot_data['service_amount_net_count'].max(),
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
    ax2.set_title(f'Топ-{top_n} кодов услуг по количеству', fontsize=24, fontweight='bold', pad=20)
    
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
    eda_style.save_figure(fig2, 'eda_results/unfiltered/figures/aggregation/top_services_by_count.png')
    
    # 3. Create pie chart for top 10 services by payment
    fig3, ax3 = eda_style.create_custom_figure(figsize=(14, 14))
    
    # Get top 10 services and group others
    top_10 = service_data_by_sum.head(10).copy()
    others_sum = service_data_by_sum.iloc[10:]['service_amount_net_sum'].sum()
    others_count = service_data_by_sum.iloc[10:]['service_amount_net_count'].sum()
    
    # Add "Others" category
    top_10.loc[len(top_10)] = {
        'service_code': 'Другие',
        'service_amount_net_sum': others_sum,
        'service_amount_net_mean': 0,  # Not relevant
        'service_amount_net_count': others_count,
        'service_document_id_nunique': 0,  # Not relevant
        'client_id_nunique': 0  # Not relevant
    }
    
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
        top_10['service_amount_net_sum'], 
        labels=None,
        autopct='',
        startangle=90,
        colors=pie_colors
    )
    
    # Add legend with percentages and counts
    total_payment = top_10['service_amount_net_sum'].sum()
    labels = [f"{code}: {eda_style.format_currency(payment, None)} руб. ({payment/total_payment:.1%}, {count:,} услуг)" 
              for code, payment, count in zip(top_10['service_code'], top_10['service_amount_net_sum'], top_10['service_amount_net_count'])]
    
    ax3.legend(wedges, labels, title="Коды услуг", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14)
    
    # Set title
    ax3.set_title('Распределение выплат по кодам услуг (Топ-10)', fontsize=24, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    eda_style.save_figure(fig3, 'eda_results/unfiltered/figures/aggregation/service_code_pie.png')
    
    return fig1, fig2, fig3
    
def plot_verification(monthly_data):
    """Create a verification plot that shows the relationship between calculated metrics"""
    print("\n=== ВЕРИФИКАЦИЯ РАСЧЕТОВ ===")
    
    # Create figure
    fig, ax = eda_style.create_custom_figure(figsize=(16, 8))
    
    # Calculate and plot direct sum
    ax.plot(
        monthly_data['date'],
        monthly_data['service_amount_net_sum'],
        'o-',
        color=eda_style.COLORS['blue'],
        linewidth=3,
        label='Прямая сумма выплат'
    )
    
    # Calculate and plot product of average and count
    product = monthly_data['avg_payment_per_service'] * monthly_data['service_amount_net_count']
    ax.plot(
        monthly_data['date'],
        product,
        's--',
        color=eda_style.COLORS['red'],
        linewidth=2,
        label='Произведение среднего на количество'
    )
    
    # Calculate difference percentage
    diff_pct = (abs(product - monthly_data['service_amount_net_sum']) / monthly_data['service_amount_net_sum'] * 100)
    max_diff = diff_pct.max()
    
    # Format axes
    ax.set_xlabel('Месяц', fontsize=16, fontweight='bold')
    ax.set_ylabel('Сумма выплат (руб.)', fontsize=16, fontweight='bold')
    ax.set_title(f'Верификация расчетов: Прямая сумма vs Произведение (макс. разница {max_diff:.6f}%)', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Format x-axis with month-year labels
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    
    # Format y-axis with currency
    ax.yaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=14, frameon=True, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/aggregation/verification_plot.png')
    
    print(f"Максимальная разница между прямой суммой и произведением: {max_diff:.6f}%")
    
    return fig

if __name__ == "__main__":
    # Load raw data directly from CSV
    data = load_raw_data()
    
    if data is not None:
        # Perform monthly aggregations
        monthly_data = monthly_aggregations(data)
        monthly_data.to_csv('eda_results/unfiltered/data/monthly_aggregation.csv', index=False)
        
        # Perform service code aggregations
        service_data_by_sum, service_data_by_count = service_code_aggregations(data)
        
        # Create monthly visualizations
        plot_monthly_service_metrics(monthly_data)
        plot_monthly_client_metrics(monthly_data)
        plot_monthly_document_metrics(monthly_data)
        
        # Create service code visualizations
        plot_top_service_codes(service_data_by_sum, service_data_by_count)
        
        # Verify calculations
        plot_verification(monthly_data)
        
        print("\nАнализ агрегаций успешно завершен. Результаты сохранены в 'eda_results/unfiltered'.") 