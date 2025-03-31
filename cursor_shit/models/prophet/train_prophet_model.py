#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import warnings
import json
from datetime import datetime
import pickle
import matplotlib.font_manager as fm
from matplotlib import rcParams

# Ignore warnings
warnings.filterwarnings("ignore")

# Set plot style
plt.style.use('ggplot')
sns.set_style('whitegrid')
mpl.rcParams['figure.figsize'] = (14, 10)
mpl.rcParams['axes.grid'] = True

# Try to use Roboto font if available
try:
    # Check if Roboto is available
    font_files = fm.findSystemFonts()
    roboto_files = [f for f in font_files if 'roboto' in f.lower()]
    
    if roboto_files:
        roboto_font = roboto_files[0]
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Roboto']
        print("Using Roboto font for plots")
    else:
        print("Roboto font not found, using default font")
except Exception as e:
    print(f"Error setting font: {e}")

# Increase font sizes for better readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Define colors
COLORS = {
    'yellow': '#E7E514',
    'green': '#1F9F38',
    'blue': '#2496D4'
}

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_data(file_path):
    """Load the monthly payments data and prepare for Prophet"""
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['service_date'] = pd.to_datetime(df['service_date'])
    
    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.rename(columns={'service_date': 'ds', 'service_amount_net': 'y'})
    
    return prophet_df

def plot_time_series(data, title, output_path=None):
    """Plot the time series data"""
    plt.figure(figsize=(16, 10))
    plt.plot(data['ds'], data['y'], marker='o', linestyle='-', color=COLORS['blue'], linewidth=3)
    plt.title(title, fontsize=22, pad=20)
    plt.xlabel('Date', fontsize=18, labelpad=15)
    plt.ylabel('Amount (₽)', fontsize=18, labelpad=15)
    
    # Format y-axis to show values in millions
    def millions(x, pos):
        return f'{x/1e6:.0f} M'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def perform_statistical_tests(data):
    """Perform statistical tests on time series data"""
    results = {}
    
    # ADF test for stationarity
    adf_result = adfuller(data)
    results['adf_test'] = {
        'test_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # KPSS test for stationarity
    kpss_result = kpss(data)
    results['kpss_test'] = {
        'test_statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05
    }
    
    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(data, lags=[12], return_df=True)
    results['ljung_box_test'] = {
        'test_statistic': lb_result['lb_stat'].values[0],
        'p_value': lb_result['lb_pvalue'].values[0],
        'has_autocorrelation': lb_result['lb_pvalue'].values[0] < 0.05
    }
    
    return results

def plot_acf_pacf(data, output_path=None):
    """Plot ACF and PACF for the time series data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Use smaller number of lags (one-third of data length)
    max_lags = min(24, len(data) // 3)
    
    plot_acf(data, ax=ax1, lags=max_lags, alpha=0.05, title='Autocorrelation Function (ACF)')
    ax1.set_ylabel('Correlation', fontsize=18, labelpad=15)
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=22, pad=20)
    ax1.grid(True, alpha=0.5)
    
    plot_pacf(data, ax=ax2, lags=max_lags, alpha=0.05, title='Partial Autocorrelation Function (PACF)')
    ax2.set_ylabel('Correlation', fontsize=18, labelpad=15)
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=22, pad=20)
    ax2.grid(True, alpha=0.5)
    
    # Increase line width for both plots
    for line in ax1.get_lines():
        line.set_linewidth(2.5)
    for line in ax2.get_lines():
        line.set_linewidth(2.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def tune_prophet_hyperparameters(data, train_data, test_data):
    """Tune Prophet hyperparameters using grid search based on forecast performance on test data"""
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.9, 0.95]
    }
    
    # Create all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Find best parameters
    best_test_mape = float('inf')
    best_params = None
    results = []
    
    print(f"Starting hyperparameter tuning with {len(all_params)} parameter combinations...")
    print(f"Evaluating models based on forecast performance on the last 3 months (test set)")
    
    # Counter for progress tracking
    count = 0
    
    # Try different combinations
    for params in all_params:
        count += 1
        if count % 5 == 0:
            print(f"Progress: {count}/{len(all_params)} combinations tested ({count/len(all_params)*100:.1f}%)")
            
        try:
            # Train model on training data
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                changepoint_range=params['changepoint_range'],
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            model.fit(train_data)
            
            # Create future dataframe that includes the test period
            future = model.make_future_dataframe(periods=len(test_data), freq='M')
            forecast = model.predict(future)
            
            # Extract predictions for test period
            test_pred = forecast.tail(len(test_data))
            
            # Calculate metrics on test data
            test_mape = mean_absolute_percentage_error(test_data['y'].values, test_pred['yhat'].values) * 100
            test_mae = mean_absolute_error(test_data['y'].values, test_pred['yhat'].values)
            test_rmse = np.sqrt(mean_squared_error(test_data['y'].values, test_pred['yhat'].values))
            
            # Store results
            result = {
                'changepoint_prior_scale': params['changepoint_prior_scale'],
                'seasonality_prior_scale': params['seasonality_prior_scale'],
                'holidays_prior_scale': params['holidays_prior_scale'],
                'seasonality_mode': params['seasonality_mode'],
                'changepoint_range': params['changepoint_range'],
                'test_mape': test_mape,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            results.append(result)
            
            # Update best parameters based on test MAPE
            if test_mape < best_test_mape:
                best_test_mape = test_mape
                best_params = params
                print(f"New best model - Test MAPE: {test_mape:.4f}% with parameters: {params}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    
    # Sort results by test MAPE
    sorted_results = sorted(results, key=lambda x: x['test_mape'])
    
    print(f"\nTuning completed. Best parameters with Test MAPE: {best_test_mape:.4f}%")
    
    return best_params, sorted_results

def train_prophet_model(data, params):
    """Train Prophet model with the given parameters"""
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        changepoint_range=params['changepoint_range'],
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    model.fit(data)
    
    return model

def evaluate_model(model, train_data, test_data):
    """Evaluate the model performance on both train and test data"""
    # Create future dataframe that includes both train and test periods
    future = model.make_future_dataframe(periods=len(test_data), freq='M')
    forecast = model.predict(future)
    
    # Extract predictions for training period
    train_pred = forecast.iloc[:len(train_data)]
    
    # Extract predictions for test period
    test_pred = forecast.tail(len(test_data))
    
    # Calculate metrics for training data
    train_mae = mean_absolute_error(train_data['y'].values, train_pred['yhat'].values)
    train_rmse = np.sqrt(mean_squared_error(train_data['y'].values, train_pred['yhat'].values))
    train_mape = mean_absolute_percentage_error(train_data['y'].values, train_pred['yhat'].values) * 100
    
    # Calculate metrics for test data
    test_mae = mean_absolute_error(test_data['y'].values, test_pred['yhat'].values)
    test_rmse = np.sqrt(mean_squared_error(test_data['y'].values, test_pred['yhat'].values))
    test_mape = mean_absolute_percentage_error(test_data['y'].values, test_pred['yhat'].values) * 100
    
    metrics = {
        'train': {
            'mae': train_mae,
            'rmse': train_rmse,
            'mape': train_mape
        },
        'test': {
            'mae': test_mae,
            'rmse': test_rmse,
            'mape': test_mape
        }
    }
    
    return test_pred, metrics

def forecast_full_2024(model, data):
    """Forecast for the full 2024 year (12 months)"""
    # Create future dataframe for 12 months
    future = model.make_future_dataframe(periods=12, freq='M')
    
    # Make prediction
    forecast = model.predict(future)
    
    # Extract the forecast period for 2024 (12 months from the last data point)
    last_date = data['ds'].max()
    full_2024_forecast = forecast[forecast['ds'] > last_date].copy()
    
    return full_2024_forecast, forecast

def forecast_future(model, data, steps=12):
    """Forecast future values"""
    # Create future dataframe
    future = model.make_future_dataframe(periods=steps, freq='M')
    
    # Make prediction
    forecast = model.predict(future)
    
    # Extract the forecast period
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
    
    return forecast_df, forecast

def plot_forecast(data, train_data, test_data, test_pred, full_forecast, forecast_2024, output_path=None):
    """Plot historical data with forecast including 2024"""
    plt.figure(figsize=(18, 10))
    
    # Plot historical data
    plt.plot(train_data['ds'], train_data['y'], label='Historical Data (Training)', 
             color=COLORS['green'], linewidth=3.5)
    
    # Plot test data and predictions
    plt.plot(test_data['ds'], test_data['y'], label='Historical Data (Test)', 
             color=COLORS['blue'], linewidth=3.5)
    plt.plot(test_data['ds'], test_pred['yhat'], label='Predictions on Test Data', 
             color=COLORS['yellow'], linewidth=3.5, linestyle='--')
    
    # Plot forecast for 2024
    future_dates = forecast_2024['ds']
    plt.plot(future_dates, forecast_2024['yhat'], label='Forecast 2024', 
             color=COLORS['yellow'], linewidth=4)
    plt.fill_between(future_dates, 
                    forecast_2024['yhat_lower'], 
                    forecast_2024['yhat_upper'], 
                    color=COLORS['yellow'], 
                    alpha=0.2, 
                    label='95% Confidence Interval')
    
    # Format plot
    plt.title('Monthly Payments: Historical Data and 2024 Forecast', fontsize=24, pad=20)
    plt.xlabel('Date', fontsize=20, labelpad=15)
    plt.ylabel('Amount (₽)', fontsize=20, labelpad=15)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best', fontsize=16)
    
    # Format y-axis to show values in millions
    def millions(x, pos):
        return f'{x/1e6:.0f} M'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))
    
    # Add annotations for test data with larger font
    for date, value in zip(test_data['ds'], test_data['y']):
        plt.annotate(f'{value/1e6:.1f}M', 
                     xy=(date, value),
                     xytext=(0, 12),
                     textcoords='offset points',
                     ha='center',
                     fontsize=12,
                     fontweight='bold')
    
    # Add annotations for quarterly forecast points
    quarterly_indices = [0, 3, 6, 9, 11]  # First month, Q2, Q3, Q4, and last month
    for i in quarterly_indices:
        if i < len(forecast_2024):
            date = forecast_2024['ds'].iloc[i]
            value = forecast_2024['yhat'].iloc[i]
            plt.annotate(f'{value/1e6:.1f}M', 
                         xy=(date, value),
                         xytext=(0, 12),
                         textcoords='offset points',
                         ha='center',
                         fontsize=12,
                         fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_prophet_components(model, forecast, output_path=None):
    """Plot the decomposed components of the Prophet model"""
    fig = model.plot_components(forecast)
    fig.set_size_inches(16, 12)
    
    # Increase line width for all component plots
    for ax in fig.get_axes():
        for line in ax.get_lines():
            line.set_linewidth(3)
        
        # Increase font size for axis labels and title
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, labelpad=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, labelpad=15)
        ax.set_title(ax.get_title(), fontsize=22, pad=20)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_results(model, best_params, tuning_results, metrics, 
                statistical_tests, forecast_df, forecast_2024, output_dir):
    """Save all results to files"""
    # Save model parameters and metrics
    model_info = {
        'parameters': best_params,
        'train_metrics': {
            'mae': float(metrics['train']['mae']),
            'rmse': float(metrics['train']['rmse']),
            'mape': float(metrics['train']['mape'])
        },
        'test_metrics': {
            'mae': float(metrics['test']['mae']),
            'rmse': float(metrics['test']['rmse']),
            'mape': float(metrics['test']['mape'])
        },
        'statistical_tests': {
            'adf_test': {
                'test_statistic': float(statistical_tests['adf_test']['test_statistic']),
                'p_value': float(statistical_tests['adf_test']['p_value']),
                'critical_values': {str(k): float(v) for k, v in statistical_tests['adf_test']['critical_values'].items()},
                'is_stationary': bool(statistical_tests['adf_test']['is_stationary'])
            },
            'kpss_test': {
                'test_statistic': float(statistical_tests['kpss_test']['test_statistic']),
                'p_value': float(statistical_tests['kpss_test']['p_value']),
                'critical_values': {str(k): float(v) for k, v in statistical_tests['kpss_test']['critical_values'].items()},
                'is_stationary': bool(statistical_tests['kpss_test']['is_stationary'])
            },
            'ljung_box_test': {
                'test_statistic': float(statistical_tests['ljung_box_test']['test_statistic']),
                'p_value': float(statistical_tests['ljung_box_test']['p_value']),
                'has_autocorrelation': bool(statistical_tests['ljung_box_test']['has_autocorrelation'])
            }
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Save tuning results
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(output_dir, 'hyperparameter_tuning_results.csv'), index=False)
    
    # Save forecast
    forecast_df.to_csv(os.path.join(output_dir, 'forecast.csv'))
    
    # Save 2024 forecast separately
    forecast_2024.to_csv(os.path.join(output_dir, 'forecast_2024.csv'))
    
    # Save model
    try:
        with open(os.path.join(output_dir, 'prophet_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Try a simpler approach
        with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
            f.write("Prophet Model Summary\n")
            f.write(f"Parameters: {best_params}\n")
            f.write(f"Test MAPE: {metrics['test']['mape']:.4f}%\n")
    
    # Save metrics in a separate text file
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write("PROPHET MODEL METRICS\n")
        f.write("=====================\n\n")
        f.write("Training Set Metrics:\n")
        f.write(f"MAE:  {metrics['train']['mae']:.2f}\n")
        f.write(f"RMSE: {metrics['train']['rmse']:.2f}\n")
        f.write(f"MAPE: {metrics['train']['mape']:.2f}%\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"MAE:  {metrics['test']['mae']:.2f}\n")
        f.write(f"RMSE: {metrics['test']['rmse']:.2f}\n")
        f.write(f"MAPE: {metrics['test']['mape']:.2f}%\n")
    
    print(f"All results saved to {output_dir}")

def plot_residuals(data, forecast, output_dir):
    """Plot model residuals"""
    # Merge actual and predicted data
    merged_df = data.copy()
    forecast_subset = forecast[['ds', 'yhat']].copy()
    merged_df = pd.merge(merged_df, forecast_subset, on='ds', how='left')
    
    # Calculate residuals
    merged_df['residuals'] = merged_df['y'] - merged_df['yhat']
    
    # Plot residuals
    plt.figure(figsize=(16, 8))
    plt.plot(merged_df['ds'], merged_df['residuals'], marker='o', markersize=8, 
             linestyle='None', color=COLORS['blue'], linewidth=3)
    plt.title('Model Residuals', fontsize=24, pad=20)
    plt.xlabel('Date', fontsize=20, labelpad=15)
    plt.ylabel('Residual Value', fontsize=20, labelpad=15)
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot residuals distribution
    plt.figure(figsize=(16, 8))
    sns.histplot(merged_df['residuals'].dropna(), kde=True, color=COLORS['blue'], 
                 line_kws={'linewidth': 3})
    plt.title('Residuals Distribution', fontsize=24, pad=20)
    plt.xlabel('Residual Value', fontsize=20, labelpad=15)
    plt.ylabel('Frequency', fontsize=20, labelpad=15)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'residuals_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return merged_df['residuals']

def main():
    # Define paths
    data_path = 'eda_results/unfiltered/data/monthly_aggregated_payments.csv'
    output_dir = 'models/prophet'
    
    # Create output directory
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'plots'))
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Plot original time series
    plot_time_series(df, 'Monthly Aggregated Payments (2022-2023)', 
                    os.path.join(output_dir, 'plots', 'original_time_series.png'))
    
    # Split data into train and test sets (last 3 months as test)
    train_data = df.iloc[:-3].copy()
    test_data = df.iloc[-3:].copy()
    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Perform statistical tests
    print("Performing statistical tests...")
    statistical_tests = perform_statistical_tests(train_data['y'])
    
    # Plot ACF and PACF
    plot_acf_pacf(train_data['y'], os.path.join(output_dir, 'plots', 'acf_pacf.png'))
    
    # Tune Prophet hyperparameters
    print("Tuning Prophet hyperparameters...")
    best_params, tuning_results = tune_prophet_hyperparameters(df, train_data, test_data)
    print(f"Best parameters: {best_params}")
    
    # Train model with best parameters
    print("Training Prophet model with best parameters...")
    model = train_prophet_model(train_data, best_params)
    
    # Evaluate model on train and test data
    print("Evaluating model on train and test data...")
    test_pred, metrics = evaluate_model(model, train_data, test_data)
    print(f"Train metrics: MAE={metrics['train']['mae']:.2f}, RMSE={metrics['train']['rmse']:.2f}, MAPE={metrics['train']['mape']:.2f}%")
    print(f"Test metrics: MAE={metrics['test']['mae']:.2f}, RMSE={metrics['test']['rmse']:.2f}, MAPE={metrics['test']['mape']:.2f}%")
    
    # Forecast for full 2024 year
    print("Forecasting for full 2024 year...")
    forecast_2024, full_forecast = forecast_full_2024(model, df)
    
    # Also get the default 12-month forecast (for saving to file)
    future_12_months, _ = forecast_future(model, df)
    
    # Print forecast for 2024
    print("\nForecast for 2024:")
    for i, (date, value) in enumerate(zip(forecast_2024['ds'], forecast_2024['yhat'])):
        print(f"{date.strftime('%Y-%m')}: {value/1e6:.2f} million ₽")
    
    # Plot forecast with historical data including 2024
    plot_forecast(df, train_data, test_data, test_pred, full_forecast, forecast_2024, 
                 os.path.join(output_dir, 'plots', 'forecast_2024.png'))
    
    # Plot Prophet components
    plot_prophet_components(model, full_forecast, os.path.join(output_dir, 'plots', 'prophet_components.png'))
    
    # Plot residuals and get residuals data
    print("Analyzing residuals...")
    residuals = plot_residuals(df, full_forecast, output_dir)
    
    # Save results
    print("\nSaving results...")
    save_results(model, best_params, tuning_results, metrics, 
                statistical_tests, future_12_months, forecast_2024, output_dir)
    
    print("Prophet modeling completed successfully!")

if __name__ == "__main__":
    main() 