#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    """Load the monthly payments data"""
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['service_date'] = pd.to_datetime(df['service_date'])
    df.set_index('service_date', inplace=True)
    return df

def plot_time_series(data, title, output_path=None):
    """Plot the time series data"""
    plt.figure(figsize=(16, 10))
    plt.plot(data.index, data['service_amount_net'], marker='o', linestyle='-', color=COLORS['blue'], linewidth=3)
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

def tune_sarima_parameters(data, train_data, test_data):
    """Tune SARIMA parameters using grid search based on forecast performance on test data"""
    # Define parameter grid
    p = range(0, 3)
    d = range(0, 2)  # Reduced range
    q = range(0, 3)
    P = range(0, 2)
    D = range(0, 2)
    Q = range(0, 2)
    s = [12]  # Monthly seasonality
    
    # Create parameter combinations
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in itertools.product(P, D, Q, s)]
    
    # Find best parameters
    best_test_mape = float('inf')
    best_params = None
    best_seasonal_params = None
    results = []
    
    print(f"Starting hyperparameter tuning with {len(pdq)} non-seasonal and {len(seasonal_pdq)} seasonal combinations...")
    print(f"Evaluating models based on forecast performance on the last 3 months (test set)")
    total_combinations = len(pdq) * len(seasonal_pdq)
    print(f"Total parameter combinations to test: {total_combinations}")
    
    # Counter for progress tracking
    count = 0
    
    # Try different combinations
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total_combinations} combinations tested ({count/total_combinations*100:.1f}%)")
                
            try:
                # Train model on training data
                model = SARIMAX(train_data,
                               order=param,
                               seasonal_order=seasonal_param,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                
                model_fit = model.fit(disp=False, maxiter=50)  # Limit iterations for speed
                
                # Make predictions on test data (last 3 months)
                start = len(train_data)
                end = len(train_data) + len(test_data) - 1
                test_pred = model_fit.get_forecast(steps=len(test_data)).predicted_mean
                
                # Calculate metrics on test data
                test_mape = mean_absolute_percentage_error(test_data.values, test_pred.values) * 100
                test_mae = mean_absolute_error(test_data.values, test_pred.values)
                test_rmse = np.sqrt(mean_squared_error(test_data.values, test_pred.values))
                
                # Calculate in-sample metrics for reference
                in_sample_pred = model_fit.predict(dynamic=False)
                in_sample_mape = mean_absolute_percentage_error(train_data, in_sample_pred) * 100
                
                # Store results
                result = {
                    'order': str(param),
                    'seasonal_order': str(seasonal_param),
                    'test_mape': test_mape,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'in_sample_mape': in_sample_mape,
                    'aic': model_fit.aic,
                    'bic': model_fit.bic
                }
                results.append(result)
                
                # Update best parameters based on test MAPE
                if test_mape < best_test_mape:
                    best_test_mape = test_mape
                    best_params = param
                    best_seasonal_params = seasonal_param
                    print(f'New best model: SARIMA{param}x{seasonal_param} - Test MAPE: {test_mape:.4f}%')
                
            except Exception as e:
                continue
    
    # Sort results by test MAPE
    sorted_results = sorted(results, key=lambda x: x['test_mape'])
    
    print(f"\nTuning completed. Best model: SARIMA{best_params}x{best_seasonal_params} with Test MAPE: {best_test_mape:.4f}%")
    
    return best_params, best_seasonal_params, sorted_results

def train_sarima_model(data, order, seasonal_order):
    """Train SARIMA model with the given parameters"""
    model = SARIMAX(data,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    
    return model_fit

def evaluate_model(model, train_data, test_data):
    """Evaluate the model performance on both train and test data"""
    # Get in-sample predictions for training data
    train_pred = model.predict(start=0, end=len(train_data)-1, dynamic=False)
    
    # Get start and end dates for test predictions
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    
    # Make predictions on test data
    test_pred = model.predict(start=start, end=end, dynamic=True)
    
    # Calculate metrics for training data
    train_mae = mean_absolute_error(train_data, train_pred)
    train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))
    train_mape = mean_absolute_percentage_error(train_data, train_pred) * 100
    
    # Calculate metrics for test data
    test_mae = mean_absolute_error(test_data, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))
    test_mape = mean_absolute_percentage_error(test_data, test_pred) * 100
    
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

def forecast_future(model, data, steps=12):
    """Forecast future values"""
    # Re-fit the model on the full dataset
    full_model = SARIMAX(data, 
                         order=model.specification['order'],
                         seasonal_order=model.specification['seasonal_order'],
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    full_model_fit = full_model.fit(disp=False)
    
    # Get the forecast
    forecast = full_model_fit.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    # Create date range for forecast
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=pd.Timestamp(last_date) + pd.DateOffset(months=1), 
                                  periods=steps, 
                                  freq='M')
    
    # Create DataFrame with forecast
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    }, index=forecast_dates)
    
    return forecast_df, full_model_fit

def forecast_full_2024(model, data):
    """Forecast for the full 2024 year (12 months)"""
    # Get the forecast for the next 12 months
    forecast_df, full_model_fit = forecast_future(model, data, steps=12)
    
    # Ensure the forecast covers 2024
    forecast_2024 = forecast_df.copy()
    
    return forecast_2024, full_model_fit

def plot_forecast(data, train_data, test_data, test_pred, forecast_df, output_path=None):
    """Plot historical data with forecast for 2024"""
    plt.figure(figsize=(18, 10))
    
    # Plot historical data
    plt.plot(train_data.index, train_data, label='Historical Data (Training)', 
             color=COLORS['green'], linewidth=3.5)
    
    # Plot test data and predictions
    plt.plot(test_data.index, test_data, label='Historical Data (Test)', 
             color=COLORS['blue'], linewidth=3.5)
    plt.plot(test_data.index, test_pred, label='Predictions on Test Data', 
             color=COLORS['yellow'], linewidth=3.5, linestyle='--')
    
    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['forecast'], label='2024 Forecast', 
             color=COLORS['yellow'], linewidth=4)
    plt.fill_between(forecast_df.index, 
                    forecast_df['lower_ci'], 
                    forecast_df['upper_ci'], 
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
    for date, value in zip(test_data.index, test_data.values):
        plt.annotate(f'{value/1e6:.1f}M', 
                     xy=(date, value),
                     xytext=(0, 12),
                     textcoords='offset points',
                     ha='center',
                     fontsize=12,
                     fontweight='bold')
    
    # Add annotations for quarterly forecast points
    quarterly_indices = [0, 3, 6, 9, 11]  # First month, Q2, Q3, Q4, and last month
    forecast_dates = forecast_df.index
    forecast_values = forecast_df['forecast'].values
    
    for i in quarterly_indices:
        if i < len(forecast_df):
            date = forecast_dates[i]
            value = forecast_values[i]
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

def save_results(model, best_params, best_seasonal_params, tuning_results, metrics, 
                statistical_tests, forecast_df, output_dir):
    """Save all results to files"""
    # Save model parameters and metrics
    model_info = {
        'order': str(best_params),
        'seasonal_order': str(best_seasonal_params),
        'aic': float(model.aic),
        'bic': float(model.bic),
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
    forecast_df.to_csv(os.path.join(output_dir, 'forecast_2024.csv'))
    
    # Save model
    try:
        with open(os.path.join(output_dir, 'sarima_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Try a simpler approach
        with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
            f.write(str(model.summary()))
            f.write("\n\nBest parameters:\n")
            f.write(f"Order: {best_params}\n")
            f.write(f"Seasonal order: {best_seasonal_params}\n")
    
    # Save metrics in a separate text file
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write("SARIMA MODEL METRICS\n")
        f.write("====================\n\n")
        f.write("Training Set Metrics:\n")
        f.write(f"MAE:  {metrics['train']['mae']:.2f}\n")
        f.write(f"RMSE: {metrics['train']['rmse']:.2f}\n")
        f.write(f"MAPE: {metrics['train']['mape']:.2f}%\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"MAE:  {metrics['test']['mae']:.2f}\n")
        f.write(f"RMSE: {metrics['test']['rmse']:.2f}\n")
        f.write(f"MAPE: {metrics['test']['mape']:.2f}%\n")
    
    print(f"All results saved to {output_dir}")

def main():
    # Define paths
    data_path = 'eda_results/unfiltered/data/monthly_aggregated_payments.csv'
    output_dir = 'models/sarima'
    
    # Create output directory
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'plots'))
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    time_series = df['service_amount_net']
    
    # Plot original time series
    plot_time_series(df, 'Monthly Aggregated Payments (2022-2023)', 
                    os.path.join(output_dir, 'plots', 'original_time_series.png'))
    
    # Split data into train and test sets (last 3 months as test)
    train_data = time_series[:-3]
    test_data = time_series[-3:]
    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Perform statistical tests
    print("Performing statistical tests...")
    statistical_tests = perform_statistical_tests(train_data)
    
    # Plot ACF and PACF
    plot_acf_pacf(train_data, os.path.join(output_dir, 'plots', 'acf_pacf.png'))
    
    # Tune SARIMA parameters based on test set performance
    print("Tuning SARIMA parameters...")
    best_params, best_seasonal_params, tuning_results = tune_sarima_parameters(time_series, train_data, test_data)
    print(f"Best parameters: SARIMA{best_params}x{best_seasonal_params}")
    
    # Train model with best parameters
    print("Training SARIMA model with best parameters...")
    model = train_sarima_model(train_data, best_params, best_seasonal_params)
    
    # Print model summary
    print(model.summary())
    
    # Save model summary to file
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model.summary()))
        f.write("\n\nModel selected based on forecast performance on the last 3 months (test set).\n")
    
    # Evaluate model on train and test data
    print("Evaluating model on train and test data...")
    test_pred, metrics = evaluate_model(model, train_data, test_data)
    print(f"Train metrics: MAE={metrics['train']['mae']:.2f}, RMSE={metrics['train']['rmse']:.2f}, MAPE={metrics['train']['mape']:.2f}%")
    print(f"Test metrics: MAE={metrics['test']['mae']:.2f}, RMSE={metrics['test']['rmse']:.2f}, MAPE={metrics['test']['mape']:.2f}%")
    
    # Forecast for 2024 (next 12 months)
    print("Forecasting for 2024...")
    forecast_df, full_model = forecast_full_2024(model, time_series)
    
    # Print forecast for 2024
    print("\nForecast for 2024:")
    for date, value in zip(forecast_df.index, forecast_df['forecast']):
        print(f"{date.strftime('%Y-%m')}: {value/1e6:.2f} million ₽")
    
    # Plot forecast with historical data
    plot_forecast(time_series, train_data, test_data, test_pred, forecast_df, 
                 os.path.join(output_dir, 'plots', 'forecast_2024.png'))
    
    # Save results
    print("\nSaving results...")
    save_results(full_model, best_params, best_seasonal_params, tuning_results, metrics, 
                statistical_tests, forecast_df, output_dir)
    
    # Additional diagnostics
    residuals = full_model.resid
    
    # Plot residuals
    plt.figure(figsize=(16, 8))
    plt.plot(residuals, marker='o', markersize=8, linestyle='None', color=COLORS['blue'], linewidth=3)
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
    sns.histplot(residuals, kde=True, color=COLORS['blue'], line_kws={'linewidth': 3})
    plt.title('Residuals Distribution', fontsize=24, pad=20)
    plt.xlabel('Residual Value', fontsize=20, labelpad=15)
    plt.ylabel('Frequency', fontsize=20, labelpad=15)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'residuals_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SARIMA modeling completed successfully!")

if __name__ == "__main__":
    main() 