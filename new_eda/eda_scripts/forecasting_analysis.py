#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time series forecasting analysis for medical services data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# For better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('eda_results/unfiltered/data', exist_ok=True)
os.makedirs('eda_results/unfiltered/figures/forecasting', exist_ok=True)

def load_monthly_data(filepath='eda_results/unfiltered/data/monthly_aggregated_payments.csv'):
    """Load the monthly aggregated dataset"""
    print(f"Loading monthly data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['service_date'] = pd.to_datetime(df['service_date'])
        return df
    except FileNotFoundError:
        try:
            # Try loading processed data directly if monthly data isn't available
            df = pd.read_csv('data/unfiltered/processed_data.csv')
            print("Monthly data not found, aggregating from unfiltered processed data...")
            df['service_date'] = pd.to_datetime(df['service_date'])
            
            # Aggregate by month
            monthly_data = df.groupby(pd.Grouper(key='service_date', freq='M'))[['service_amount_net']].sum().reset_index()
            monthly_data['year_month'] = monthly_data['service_date'].dt.strftime('%Y-%m')
            
            # Save for future use
            os.makedirs('eda_results/unfiltered/data', exist_ok=True)
            monthly_data.to_csv(filepath, index=False)
            
            return monthly_data
        except FileNotFoundError:
            print("Could not find processed data either. Run eda_scripts/eda_main.py first.")
            return None

def time_series_analysis(df):
    """Perform time series analysis and decomposition"""
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Set index to datetime for time series analysis
    ts_data = df.set_index('service_date')['service_amount_net']
    
    # Perform seasonal decomposition if we have enough data
    if len(ts_data) >= 12:  # Need at least a year of data
        print("\nPerforming Seasonal Decomposition...")
        try:
            # Decompose time series
            result = seasonal_decompose(ts_data, model='additive', period=12)
            
            # Plot decomposition
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
            
            result.observed.plot(ax=ax1)
            ax1.set_title('Observed', fontsize=16, fontweight='bold')
            
            result.trend.plot(ax=ax2)
            ax2.set_title('Trend', fontsize=16, fontweight='bold')
            
            result.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonality', fontsize=16, fontweight='bold')
            
            result.resid.plot(ax=ax4)
            ax4.set_title('Residuals', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('eda_results/unfiltered/figures/forecasting/time_series_decomposition.png', dpi=300, bbox_inches='tight')
            
            # Save decomposition components
            decomp_data = pd.DataFrame({
                'observed': result.observed,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid
            })
            decomp_data.to_csv('eda_results/unfiltered/data/time_series_decomposition.csv')
            
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
    else:
        print("Not enough data for seasonal decomposition (need at least 12 months)")
    
    # Plot average by month (seasonality)
    df['month'] = df['service_date'].dt.month
    monthly_avg = df.groupby('month')['service_amount_net'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    plt.bar(monthly_avg['month'], monthly_avg['service_amount_net']/1e6, color=sns.color_palette('viridis', 12))
    plt.title('Average Monthly Payments (Seasonality)', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Payment (Millions)', fontsize=14)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('eda_results/unfiltered/figures/forecasting/monthly_seasonality.png', dpi=300, bbox_inches='tight')
    
    return ts_data

def forecast_future_payments(ts_data, periods=6):
    """Forecast future payments using SARIMA model"""
    print("\n=== PAYMENT FORECASTING ===")
    
    if len(ts_data) < 18:  # Need sufficient data for forecasting
        print("Not enough data for reliable forecasting (need at least 18 months)")
        return
    
    try:
        print("\nTraining SARIMA model...")
        # Define SARIMA model - adjust parameters as needed based on data
        model = SARIMAX(
            ts_data, 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit model
        results = model.fit(disp=False)
        
        # Forecast future periods
        forecast = results.get_forecast(steps=periods)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Create forecast dataframe
        last_date = ts_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }, index=forecast_index)
        
        # Plot historical data and forecast
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(ts_data.index, ts_data/1e6, label='Historical Data', color='blue')
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['forecast']/1e6, label='Forecast', color='red')
        
        # Plot confidence interval
        plt.fill_between(
            forecast_df.index,
            forecast_df['lower_ci']/1e6,
            forecast_df['upper_ci']/1e6,
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        
        # Add vertical line at forecast start
        plt.axvline(x=last_date, color='green', linestyle='--', alpha=0.7, label='Forecast Start')
        
        plt.title('Payment Forecast', fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Payment Amount (Millions)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('eda_results/unfiltered/figures/forecasting/payment_forecast.png', dpi=300, bbox_inches='tight')
        
        # Save forecast data
        forecast_df.reset_index().to_csv('eda_results/unfiltered/data/payment_forecast.csv', index=False)
        
        # Print forecast summary
        print("\nForecast Summary (next 6 months):")
        print(forecast_df.reset_index()[['index', 'forecast']].rename(
            columns={'index': 'date', 'forecast': 'forecasted_amount'}
        ).to_string(index=False, float_format=lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x))
        
    except Exception as e:
        print(f"Error in forecasting: {e}")

def growth_rate_analysis(df):
    """Analyze monthly and annual growth rates"""
    print("\n=== GROWTH RATE ANALYSIS ===")
    
    # Calculate month-over-month growth
    df = df.sort_values('service_date')
    df['mom_growth'] = df['service_amount_net'].pct_change() * 100
    
    # Calculate year-over-year growth
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    
    # Create pivot for YoY comparison
    pivot = df.pivot_table(index='month', columns='year', values='service_amount_net')
    
    # Calculate YoY growth for each month
    years = sorted(df['year'].unique())
    if len(years) >= 2:
        latest_year = years[-1]
        previous_year = years[-2]
        
        if previous_year in pivot.columns and latest_year in pivot.columns:
            # Calculate YoY growth rates
            yoy_growth = ((pivot[latest_year] / pivot[previous_year]) - 1) * 100
            
            # Plot YoY growth
            plt.figure(figsize=(14, 8))
            
            # Bar chart of YoY growth by month
            plt.bar(range(1, 13), 
                   [yoy_growth.loc[m] if m in yoy_growth.index and not pd.isna(yoy_growth.loc[m]) else 0 for m in range(1, 13)],
                   color=sns.color_palette('viridis', 12))
            
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            
            plt.title(f'Year-over-Year Growth ({previous_year} to {latest_year})', fontsize=18, fontweight='bold')
            plt.xlabel('Month', fontsize=14)
            plt.ylabel('Growth Rate (%)', fontsize=14)
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add growth values as text
            for i, month in enumerate(range(1, 13)):
                if month in yoy_growth.index and not pd.isna(yoy_growth.loc[month]):
                    growth = yoy_growth.loc[month]
                    plt.text(i+1, growth + (2 if growth >= 0 else -5), 
                             f"{growth:.1f}%", ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('eda_results/unfiltered/figures/forecasting/yoy_growth.png', dpi=300, bbox_inches='tight')
            
            # Save YoY growth data
            yoy_growth.name = 'yoy_growth'
            yoy_growth.reset_index().to_csv('eda_results/unfiltered/data/yoy_growth.csv', index=False)
            
            # Print summary statistics
            print(f"\nYear-over-Year Growth ({previous_year} to {latest_year}):")
            print(f"Average monthly growth: {yoy_growth.mean():.2f}%")
            print(f"Median monthly growth: {yoy_growth.median():.2f}%")
            print(f"Minimum monthly growth: {yoy_growth.min():.2f}%")
            print(f"Maximum monthly growth: {yoy_growth.max():.2f}%")
            
            # Calculate overall annual growth
            annual_total_prev = pivot[previous_year].sum()
            annual_total_latest = pivot[latest_year].sum()
            annual_growth = ((annual_total_latest / annual_total_prev) - 1) * 100
            print(f"Overall annual growth: {annual_growth:.2f}%")
    else:
        print("Not enough years of data for year-over-year comparison")

if __name__ == "__main__":
    # Load monthly data
    monthly_data = load_monthly_data()
    
    if monthly_data is not None:
        # Perform time series analysis
        ts_data = time_series_analysis(monthly_data)
        
        # Forecast future payments
        forecast_future_payments(ts_data)
        
        # Analyze growth rates
        growth_rate_analysis(monthly_data)
        
        print("\nForecasting analysis completed successfully!") 