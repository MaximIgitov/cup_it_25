#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main EDA script for insurance payment data
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
os.makedirs('data/unfiltered', exist_ok=True)
os.makedirs('eda_results/unfiltered/data', exist_ok=True)
os.makedirs('eda_results/unfiltered/figures', exist_ok=True)

def load_data(filepath='CupIT_Sber_data.csv'):
    """
    Load the dataset and perform minimal preprocessing
    """
    print(f"Загрузка данных из {filepath}...")
    try:
        # Load data with semicolon separator
        df = pd.read_csv(filepath, sep=';')
        
        # Fix the column issue if headers have linebreaks
        if 'is_hospital' not in df.columns and any(col.endswith('is_hospital') for col in df.columns):
            # Find the problematic column
            for col in df.columns:
                if col.endswith('is_hospital'):
                    # Extract column name and fix it
                    df = df.rename(columns={col: 'is_hospital'})
                    break
        
        # Convert date columns to datetime
        if 'service_date' in df.columns:
            df['service_date'] = pd.to_datetime(df['service_date'])
        
        # Basic data description
        print("\nИнформация о датасете:")
        print(f"Всего строк: {len(df):,}")
        print(f"Диапазон дат: {df['service_date'].min().date()} - {df['service_date'].max().date()}")
        
        # Column overview
        print("\nОбзор данных:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_percent = null_count / len(df) * 100
            unique_count = df[col].nunique()
            unique_percent = unique_count / len(df) * 100
            
            dtype = df[col].dtype
            
            print(f"- {col}: {null_count:,} пропусков ({null_percent:.1f}%), "
                  f"{unique_count:,} уникальных значений ({unique_percent:.1f}%), тип {dtype}")
        
        # Check for duplicates
        dup_count = df.duplicated('service_document_id').sum()
        print(f"\nНайдено {dup_count:,} дублирующихся записей по service_document_id "
              f"({dup_count/len(df)*100:.2f}% от общего количества)")
        
        return df
    
    except FileNotFoundError:
        print(f"Файл {filepath} не найден.")
        return None

def clean_data(df):
    """
    Perform minimal cleaning without filtering out any data
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Make sure date column is datetime
    df_clean['service_date'] = pd.to_datetime(df_clean['service_date'])
    
    # Ensure correct column names and types
    if 'client_id' not in df_clean.columns and 'patient_id' in df_clean.columns:
        df_clean = df_clean.rename(columns={'patient_id': 'client_id'})
    
    if 'service_amount_net' not in df_clean.columns and 'service_amount' in df_clean.columns:
        df_clean = df_clean.rename(columns={'service_amount': 'service_amount_net'})
    
    # Ensure service_amount_net is numeric
    df_clean['service_amount_net'] = pd.to_numeric(df_clean['service_amount_net'], errors='coerce')
    
    # Save processed data
    output_path = 'data/unfiltered/processed_data.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\nСохранено {len(df_clean):,} строк в {output_path}")
    
    return df_clean

def basic_visualizations(df):
    """
    Create basic visualizations of the unfiltered data
    """
    if df is None:
        return
    
    print("\nСоздание базовых визуализаций...")
    
    # Create a figure directory if it doesn't exist
    os.makedirs('eda_results/unfiltered/figures', exist_ok=True)
    
    # 1. Monthly payment trend
    df['year_month'] = df['service_date'].dt.strftime('%Y-%m')
    monthly_totals = df.groupby('year_month')['service_amount_net'].sum().reset_index()
    monthly_totals['date'] = pd.to_datetime(monthly_totals['year_month'] + '-01')
    monthly_totals = monthly_totals.sort_values('date')
    
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot the monthly payment trend
    ax.plot(range(len(monthly_totals)), monthly_totals['service_amount_net']/1000000, 
           'o-', linewidth=3, markersize=8, color=eda_style.COLORS['blue'])
    
    # Apply common styles with Russian titles
    eda_style.apply_common_styles(
        ax,
        title='Ежемесячные страховые выплаты',
        xlabel='Месяц',
        ylabel='Объем выплат (млн руб.)'
    )
    
    # X-axis labels - use every other month for clarity
    step = 2 if len(monthly_totals) > 12 else 1
    ax.set_xticks(range(0, len(monthly_totals), step))
    ax.set_xticklabels([m for i, m in enumerate(monthly_totals['year_month']) if i % step == 0],
                     rotation=45, ha='right')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}M"))
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/monthly_payment_trend.png')
    
    # 2. Service count visualization
    service_counts = df['service_document_id'].value_counts().reset_index()
    service_counts.columns = ['service_id', 'count']
    service_counts = service_counts.sort_values('count', ascending=False)
    
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Plot top 10 services
    top_10 = service_counts.head(10)
    bars = ax.bar(range(len(top_10)), top_10['count'], 
                 color=[eda_style.COLORS['blue_grad_%d' % i] for i in range(10)])
    
    # Apply common styles with Russian titles
    eda_style.apply_common_styles(
        ax,
        title='Топ-10 страховых услуг по количеству',
        xlabel='ID услуги',
        ylabel='Количество'
    )
    
    # X-axis labels
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10['service_id'], rotation=45, ha='right')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/top_services_count.png')
    
    # 3. Payment distribution visualization
    fig, ax = eda_style.create_custom_figure(figsize=(14, 8))
    
    # Create a histogram with custom bins
    max_payment = min(df['service_amount_net'].max(), df['service_amount_net'].quantile(0.99) * 2)
    bins = np.linspace(0, max_payment, 50)
    
    # Plot the histogram
    ax.hist(df['service_amount_net'], bins=bins, 
           color=eda_style.COLORS['blue'], alpha=0.7, 
           edgecolor=eda_style.COLORS['blue_dark'])
    
    # Apply common styles with Russian titles
    eda_style.apply_common_styles(
        ax,
        title='Распределение размера страховых выплат',
        xlabel='Размер выплаты (руб.)',
        ylabel='Количество'
    )
    
    # Format x-axis with currency
    ax.xaxis.set_major_formatter(FuncFormatter(eda_style.format_currency))
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    
    # Save the figure
    eda_style.save_figure(fig, 'eda_results/unfiltered/figures/payment_distribution.png')
    
    print("Базовые визуализации созданы и сохранены в 'eda_results/unfiltered/figures'")

if __name__ == "__main__":
    # Apply professional styling with the Roboto font
    print("Применение профессиональных стилей визуализации с шрифтом Roboto...")
    
    # Load raw data
    raw_data = load_data()
    
    if raw_data is not None:
        # Clean data without filtering
        clean_df = clean_data(raw_data)
        
        # Create basic visualizations
        basic_visualizations(clean_df)
        
        print("\nОсновной EDA-скрипт успешно выполнен. Для более детального анализа запустите специализированные скрипты:")
        print("- python eda_scripts/client_analysis.py")
        print("- python eda_scripts/service_analysis.py")
        print("- python eda_scripts/temporal_analysis.py") 