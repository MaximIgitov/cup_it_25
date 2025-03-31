#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pickle
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from prophet import Prophet
from plotly.graph_objects import Layout
import plotly.figure_factory as ff

# Set Plotly default template
pio.templates.default = "plotly_white"

# Define colors - using blue to yellow spectrum for better contrast
COLORS = {
    'blue_dark': '#0055A4',
    'blue': '#2496D4',
    'cyan': '#00A6B4',
    'green': '#1F9F38',
    'lime': '#82D12E',
    'yellow': '#E7E514'
}

# Define font settings
FONT_SETTINGS = {
    'family': 'Roboto, Arial, sans-serif',
    'size': 18
}

def load_data(file_path):
    """Load the monthly payments data"""
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['service_date'] = pd.to_datetime(df['service_date'])
    
    # Create copy for Prophet (which needs ds and y columns)
    prophet_df = df.rename(columns={'service_date': 'ds', 'service_amount_net': 'y'})
    
    # Set index for SARIMA data
    df.set_index('service_date', inplace=True)
    
    return df, prophet_df

def load_saved_forecasts():
    """Load the saved forecast data from both models"""
    # Load SARIMA forecast
    sarima_forecast = pd.read_csv('models/sarima/forecast.csv')
    sarima_forecast['date'] = pd.to_datetime(sarima_forecast['Unnamed: 0'])
    sarima_forecast.drop(columns=['Unnamed: 0'], inplace=True)
    sarima_forecast.set_index('date', inplace=True)
    
    # Load Prophet forecast
    prophet_forecast = pd.read_csv('models/prophet/forecast.csv')
    prophet_forecast['date'] = pd.to_datetime(prophet_forecast['ds'])
    
    return sarima_forecast, prophet_forecast

def load_saved_metrics():
    """Load the saved metrics for both models"""
    # Load model info which contains metrics
    try:
        with open('models/sarima/model_info.json', 'r') as f:
            sarima_info = json.load(f)
        
        with open('models/prophet/model_info.json', 'r') as f:
            prophet_info = json.load(f)
            
        # Format metrics as text since we don't have the actual text files
        sarima_metrics_text = "SARIMA MODEL METRICS\n"
        sarima_metrics_text += "====================\n\n"
        
        if 'train_metrics' in sarima_info:
            sarima_metrics_text += "Training Set Metrics:\n"
            sarima_metrics_text += f"MAE:  {sarima_info['train_metrics']['mae']:.2f}\n"
            sarima_metrics_text += f"RMSE: {sarima_info['train_metrics']['rmse']:.2f}\n"
            sarima_metrics_text += f"MAPE: {sarima_info['train_metrics']['mape']:.2f}%\n\n"
        
        sarima_metrics_text += "Test Set Metrics:\n"
        sarima_metrics_text += f"MAE:  {sarima_info['test_metrics']['mae']:.2f}\n"
        sarima_metrics_text += f"RMSE: {sarima_info['test_metrics']['rmse']:.2f}\n"
        sarima_metrics_text += f"MAPE: {sarima_info['test_metrics']['mape']:.2f}%\n"
        
        prophet_metrics_text = "PROPHET MODEL METRICS\n"
        prophet_metrics_text += "=====================\n\n"
        
        if 'train_metrics' in prophet_info:
            prophet_metrics_text += "Training Set Metrics:\n"
            prophet_metrics_text += f"MAE:  {prophet_info['train_metrics']['mae']:.2f}\n"
            prophet_metrics_text += f"RMSE: {prophet_info['train_metrics']['rmse']:.2f}\n"
            prophet_metrics_text += f"MAPE: {prophet_info['train_metrics']['mape']:.2f}%\n\n"
        
        prophet_metrics_text += "Test Set Metrics:\n"
        prophet_metrics_text += f"MAE:  {prophet_info['test_metrics']['mae']:.2f}\n"
        prophet_metrics_text += f"RMSE: {prophet_info['test_metrics']['rmse']:.2f}\n"
        prophet_metrics_text += f"MAPE: {prophet_info['test_metrics']['mape']:.2f}%\n"
        
    except Exception as e:
        print(f"Warning: Could not load model metrics: {e}")
        sarima_info = {}
        prophet_info = {}
        sarima_metrics_text = "SARIMA metrics not available"
        prophet_metrics_text = "Prophet metrics not available"
    
    return sarima_metrics_text, prophet_metrics_text, sarima_info, prophet_info

def load_prophet_model():
    """Load the saved Prophet model if available"""
    try:
        with open('models/prophet/prophet_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Could not load Prophet model: {e}")
        return None

def load_sarima_model():
    """Load the saved SARIMA model if available"""
    try:
        with open('models/sarima/sarima_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Could not load SARIMA model: {e}")
        return None

def load_linear_forecast(file_path):
    """Load the linear models forecast data"""
    try:
        linear_forecast = pd.read_csv(file_path)
        linear_forecast['date'] = pd.to_datetime(linear_forecast['date'])
        return linear_forecast
    except Exception as e:
        print(f"Could not load linear models forecast: {e}")
        return None

def create_individual_model_plot(df, forecast_data, model_name, color, output_dir, months_to_show=12):
    """Create a plot for a single model forecast"""
    # Convert to millions for better display
    millions_converter = lambda x: x / 1e6
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Get last date and filter forecast data
    if isinstance(forecast_data, pd.DataFrame):
        if 'date' in forecast_data.columns:
            last_date = pd.to_datetime(forecast_data['date']).max()
            forecast_dates = pd.to_datetime(forecast_data['date'])
        else:
            last_date = forecast_data.index.max()
            forecast_dates = forecast_data.index
    else:
        return None
    
    # Filter forecast data to show only the last 12 months
    forecast_start_date = last_date - pd.DateOffset(months=months_to_show-1)
    
    # Add all historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=millions_converter(df['service_amount_net']),
        mode='lines+markers',
        name='Historical Data',
        line=dict(color=COLORS['blue'], width=4),
        marker=dict(size=10)
    ))
    
    # Add model forecast
    if model_name == 'SARIMA':
        forecast_filtered = forecast_data[forecast_data.index >= forecast_start_date]
        fig.add_trace(go.Scatter(
            x=forecast_filtered.index,
            y=millions_converter(forecast_filtered['forecast']),
            mode='lines+markers+text',
            name=f'{model_name} Forecast',
            line=dict(color=color, width=5),
            marker=dict(size=12),
            text=millions_converter(forecast_filtered['forecast']).round(2).astype(str) + ' M₽',
            textposition='top center',
            textfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']-2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=list(forecast_filtered.index) + list(forecast_filtered.index[::-1]),
            y=millions_converter(pd.concat([forecast_filtered['lower_ci'], forecast_filtered['upper_ci'][::-1]])),
            fill='toself',
            fillcolor=f'rgba{(*hex_to_rgb(color), 0.2)}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
    
    elif model_name == 'Prophet':
        forecast_filtered = forecast_data[pd.to_datetime(forecast_data['date']) >= forecast_start_date]
        prophet_dates = forecast_filtered['date'] if 'date' in forecast_filtered.columns else forecast_filtered['ds']
        
        fig.add_trace(go.Scatter(
            x=prophet_dates,
            y=millions_converter(forecast_filtered['yhat']),
            mode='lines+markers+text',
            name=f'{model_name} Forecast',
            line=dict(color=color, width=5),
            marker=dict(size=12),
            text=millions_converter(forecast_filtered['yhat']).round(2).astype(str) + ' M₽',
            textposition='top center',
            textfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']-2)
        ))
        
        # Add confidence interval
        if 'yhat_lower' in forecast_filtered.columns and 'yhat_upper' in forecast_filtered.columns:
            fig.add_trace(go.Scatter(
                x=list(prophet_dates) + list(prophet_dates[::-1]),
                y=millions_converter(pd.concat([forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'][::-1]])),
                fill='toself',
                fillcolor=f'rgba{(*hex_to_rgb(color), 0.2)}',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
    
    elif model_name == 'Linear':
        forecast_filtered = forecast_data[pd.to_datetime(forecast_data['date']) >= forecast_start_date]
        
        fig.add_trace(go.Scatter(
            x=forecast_filtered['date'],
            y=millions_converter(forecast_filtered['forecast']),
            mode='lines+markers+text',
            name=f'{model_name} Forecast',
            line=dict(color=color, width=5),
            marker=dict(size=12),
            text=millions_converter(forecast_filtered['forecast']).round(2).astype(str) + ' M₽',
            textposition='top center',
            textfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']-2)
        ))
    
    # Update layout with improved readability
    fig.update_layout(
        title={
            'text': f'{model_name} Forecast (All Historical Data + Last 12 Months of Predictions)',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(family=FONT_SETTINGS['family'], size=28)
        },
        xaxis_title='Date',
        yaxis_title='Amount (₽, millions)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SETTINGS['family'], size=20)
        ),
        font=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']),
        hovermode="x unified",
        xaxis=dict(
            tickformat="%b",  # Just month abbr without year
            tickangle=-45,
            dtick="M1",  # Show all months
            tickfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size'])
        ),
        yaxis=dict(
            tickfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']),
            gridwidth=2,
            gridcolor='rgba(220,220,220,0.8)',
            showgrid=True
        ),
        height=800,
        width=1200,
        margin=dict(t=150)  # Add more top margin for title
    )
    
    # Save as HTML and as a static PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f"{output_dir}/{model_name.lower()}_forecast.html")
    fig.write_image(f"{output_dir}/{model_name.lower()}_forecast.png", scale=2)
    
    return fig

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple for plotly"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_comparison_plot(df, sarima_forecast, prophet_forecast, linear_forecast, output_dir, months_to_show=12):
    """Create a comparison plot of all forecasts using Plotly"""
    # Convert to millions for better display
    millions_converter = lambda x: x / 1e6
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Determine the last date in the forecasts
    last_date = max(sarima_forecast.index.max(), 
                   pd.to_datetime(prophet_forecast['date']).max(),
                   pd.to_datetime(linear_forecast['date']).max() if linear_forecast is not None else pd.Timestamp('1900-01-01'))
    
    # Filter forecast data to show only the last 12 months
    forecast_start_date = last_date - pd.DateOffset(months=months_to_show-1)
    
    # Filter only forecast data, keep all historical data
    sarima_filtered = sarima_forecast[sarima_forecast.index >= forecast_start_date]
    prophet_filtered = prophet_forecast[pd.to_datetime(prophet_forecast['date']) >= forecast_start_date]
    if linear_forecast is not None:
        linear_filtered = linear_forecast[pd.to_datetime(linear_forecast['date']) >= forecast_start_date]
    
    # Add all historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=millions_converter(df['service_amount_net']),
        mode='lines+markers',
        name='Historical Data',
        line=dict(color=COLORS['blue'], width=4),
        marker=dict(size=10)
    ))
    
    # Add SARIMA forecast (last 12 months only)
    fig.add_trace(go.Scatter(
        x=sarima_filtered.index,
        y=millions_converter(sarima_filtered['forecast']),
        mode='lines+markers',
        name='SARIMA Forecast',
        line=dict(color=COLORS['green'], width=5),
        marker=dict(size=10)
    ))
    
    # Add SARIMA confidence interval
    fig.add_trace(go.Scatter(
        x=list(sarima_filtered.index) + list(sarima_filtered.index[::-1]),
        y=millions_converter(pd.concat([sarima_filtered['lower_ci'], sarima_filtered['upper_ci'][::-1]])),
        fill='toself',
        fillcolor=f'rgba{(*hex_to_rgb(COLORS["green"]), 0.2)}',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    # Add Prophet forecast (last 12 months only)
    prophet_dates = prophet_filtered['date'] if 'date' in prophet_filtered.columns else prophet_filtered['ds']
    
    fig.add_trace(go.Scatter(
        x=prophet_dates,
        y=millions_converter(prophet_filtered['yhat']),
        mode='lines+markers',
        name='Prophet Forecast',
        line=dict(color=COLORS['cyan'], width=5),
        marker=dict(size=10)
    ))
    
    # Add Prophet confidence interval
    if 'yhat_lower' in prophet_filtered.columns and 'yhat_upper' in prophet_filtered.columns:
        fig.add_trace(go.Scatter(
            x=list(prophet_dates) + list(prophet_dates[::-1]),
            y=millions_converter(pd.concat([prophet_filtered['yhat_lower'], prophet_filtered['yhat_upper'][::-1]])),
            fill='toself',
            fillcolor=f'rgba{(*hex_to_rgb(COLORS["cyan"]), 0.2)}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Add Linear Models forecast if available (last 12 months only)
    if linear_forecast is not None:
        fig.add_trace(go.Scatter(
            x=linear_filtered['date'],
            y=millions_converter(linear_filtered['forecast']),
            mode='lines+markers',
            name='Linear Models Forecast',
            line=dict(color=COLORS['yellow'], width=5),
            marker=dict(size=10)
        ))
    
    # Update layout with enhanced readability
    fig.update_layout(
        title={
            'text': 'Comparison of Forecasts (All Historical Data + Last 12 Months of Predictions)',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(family=FONT_SETTINGS['family'], size=28)
        },
        xaxis_title='Date',
        yaxis_title='Amount (₽, millions)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SETTINGS['family'], size=20)
        ),
        font=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']),
        hovermode="x unified",
        xaxis=dict(
            tickformat="%b",  # Just month abbr without year
            tickangle=-45,
            dtick="M1",  # Show all months
            tickfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size'])
        ),
        yaxis=dict(
            tickfont=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']),
            gridwidth=2,
            gridcolor='rgba(220,220,220,0.8)',
            showgrid=True
        ),
        height=800,
        width=1200,
        margin=dict(t=150)  # Add more top margin for title
    )
    
    # Save as HTML and as a static PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f"{output_dir}/forecast_comparison.html")
    fig.write_image(f"{output_dir}/forecast_comparison.png", scale=2)
    
    return fig

def create_metrics_comparison_plot(sarima_info, prophet_info, output_dir):
    """Create a comparison of metrics between SARIMA and Prophet"""
    # Extract metrics
    metrics = {
        'Model': ['SARIMA', 'SARIMA', 'Prophet', 'Prophet'],
        'Dataset': ['Train', 'Test', 'Train', 'Test'],
        'MAE': [
            sarima_info.get('train_metrics', {}).get('mae', 0),
            sarima_info.get('test_metrics', {}).get('mae', 0),
            prophet_info.get('train_metrics', {}).get('mae', 0),
            prophet_info.get('test_metrics', {}).get('mae', 0)
        ],
        'RMSE': [
            sarima_info.get('train_metrics', {}).get('rmse', 0),
            sarima_info.get('test_metrics', {}).get('rmse', 0),
            prophet_info.get('train_metrics', {}).get('rmse', 0),
            prophet_info.get('test_metrics', {}).get('rmse', 0)
        ],
        'MAPE': [
            sarima_info.get('train_metrics', {}).get('mape', 0),
            sarima_info.get('test_metrics', {}).get('mape', 0),
            prophet_info.get('train_metrics', {}).get('mape', 0),
            prophet_info.get('test_metrics', {}).get('mape', 0)
        ]
    }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Create subplots
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=("Mean Absolute Error (MAE)", 
                                      "Root Mean Squared Error (RMSE)", 
                                      "Mean Absolute Percentage Error (MAPE)"))
    
    # Add bars for each metric with updated colors
    colors = {'SARIMA': COLORS['green'], 'Prophet': COLORS['cyan']}
    patterns = {'Train': '', 'Test': '/'}  # Valid pattern shapes
    
    # Add MAE subplot
    for model in ['SARIMA', 'Prophet']:
        for dataset in ['Train', 'Test']:
            subset = metrics_df[(metrics_df['Model'] == model) & (metrics_df['Dataset'] == dataset)]
            fig.add_trace(
                go.Bar(
                    x=[f"{model} - {dataset}"],
                    y=subset['MAE'],
                    name=f"{model} - {dataset}",
                    marker_color=colors[model],
                    marker_pattern_shape=patterns[dataset],
                    showlegend=True if dataset == 'Train' else False,
                    text=subset['MAE'].round(2),
                    textposition='outside'
                ),
                row=1, col=1
            )
    
    # Add RMSE subplot
    for model in ['SARIMA', 'Prophet']:
        for dataset in ['Train', 'Test']:
            subset = metrics_df[(metrics_df['Model'] == model) & (metrics_df['Dataset'] == dataset)]
            fig.add_trace(
                go.Bar(
                    x=[f"{model} - {dataset}"],
                    y=subset['RMSE'],
                    name=f"{model} - {dataset}",
                    marker_color=colors[model],
                    marker_pattern_shape=patterns[dataset],
                    showlegend=False,
                    text=subset['RMSE'].round(2),
                    textposition='outside'
                ),
                row=1, col=2
            )
    
    # Add MAPE subplot
    for model in ['SARIMA', 'Prophet']:
        for dataset in ['Train', 'Test']:
            subset = metrics_df[(metrics_df['Model'] == model) & (metrics_df['Dataset'] == dataset)]
            fig.add_trace(
                go.Bar(
                    x=[f"{model} - {dataset}"],
                    y=subset['MAPE'],
                    name=f"{model} - {dataset}",
                    marker_color=colors[model],
                    marker_pattern_shape=patterns[dataset],
                    showlegend=False,
                    text=subset['MAPE'].round(2),
                    textposition='outside'
                ),
                row=1, col=3
            )
    
    # Update layout with Roboto font
    fig.update_layout(
        title={
            'text': 'Model Performance Comparison',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(family=FONT_SETTINGS['family'], size=28)
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SETTINGS['family'], size=20)
        ),
        font=dict(family=FONT_SETTINGS['family'], size=FONT_SETTINGS['size']),
        height=600,
        width=1200
    )
    
    # Add y-axis title to each subplot
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=3)
    
    # Save as HTML and as a static PNG
    fig.write_html(f"{output_dir}/metrics_comparison.html")
    fig.write_image(f"{output_dir}/metrics_comparison.png", scale=2)
    
    return fig

def generate_forecasts_from_models(df, prophet_df):
    """Generate forecasts by loading and using the trained models"""
    # Load the trained models
    sarima_model = load_sarima_model()
    prophet_model = load_prophet_model()
    
    # Check if models were loaded successfully
    if sarima_model is None or prophet_model is None:
        print("Warning: Could not load one or both models. Using saved forecasts instead.")
        return load_saved_forecasts()
    
    # Get the last date in the dataset
    last_date = df.index.max()
    
    # Determine the forecast period (last 3 months of 2023 + all of 2024)
    forecast_start_date = pd.Timestamp(f"2023-10-01")
    months_to_forecast = 15
    
    # Create date range for the forecast period
    forecast_dates = pd.date_range(
        start=forecast_start_date,
        periods=months_to_forecast,
        freq='M'
    )
    
    # Generate SARIMA forecast
    # Get the steps needed from the end of the data
    steps_from_data = (forecast_dates[-1] - last_date).days // 30 + 1
    
    # Handle both in-sample (historical) and out-of-sample (future) predictions
    forecast_start_idx = None
    if forecast_dates[0] <= last_date:
        # Find the index in the dataset corresponding to our forecast start date
        for i, date in enumerate(df.index):
            if date >= forecast_dates[0] or abs((date - forecast_dates[0]).days) < 15:
                forecast_start_idx = i
                break
    
    # Get in-sample predictions (for dates already in our historical data)
    if forecast_start_idx is not None:
        in_sample_pred = sarima_model.predict(start=forecast_start_idx)
        in_sample_dates = df.index[forecast_start_idx:]
        in_sample_pred = pd.Series(in_sample_pred, index=in_sample_dates)
    else:
        in_sample_pred = pd.Series(dtype=float)
    
    # Get future predictions
    if last_date < forecast_dates[-1]:
        forecast = sarima_model.get_forecast(steps=steps_from_data)
        future_pred = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=0.05)
    else:
        future_pred = pd.Series(dtype=float)
    
    # Combine both types of predictions
    all_predictions = pd.concat([in_sample_pred, future_pred])
    
    # Create confidence intervals for in-sample predictions if needed
    if 'forecast_ci' not in locals():
        forecast_ci = pd.DataFrame(index=all_predictions.index)
        forecast_ci['lower'] = all_predictions * 0.9
        forecast_ci['upper'] = all_predictions * 1.1
    
    # Create DataFrame with all predictions
    all_sarima_forecast = pd.DataFrame({
        'forecast': all_predictions,
        'lower_ci': pd.concat([in_sample_pred * 0.9, forecast_ci.iloc[:, 0] if len(forecast_ci) > 0 else pd.Series()]),
        'upper_ci': pd.concat([in_sample_pred * 1.1, forecast_ci.iloc[:, 1] if len(forecast_ci) > 0 else pd.Series()])
    })
    
    # Extract forecasts for our desired period
    sarima_forecast = pd.DataFrame(index=forecast_dates)
    
    # For dates that exist in all_sarima_forecast, use directly
    common_dates = set(forecast_dates).intersection(set(all_sarima_forecast.index))
    for date in common_dates:
        sarima_forecast.loc[date, 'forecast'] = all_sarima_forecast.loc[date, 'forecast']
        sarima_forecast.loc[date, 'lower_ci'] = all_sarima_forecast.loc[date, 'lower_ci']
        sarima_forecast.loc[date, 'upper_ci'] = all_sarima_forecast.loc[date, 'upper_ci']
    
    # For dates that don't exist, find closest date
    missing_dates = set(forecast_dates) - common_dates
    for date in missing_dates:
        closest_date = min(all_sarima_forecast.index, key=lambda x: abs(x - date))
        sarima_forecast.loc[date, 'forecast'] = all_sarima_forecast.loc[closest_date, 'forecast']
        sarima_forecast.loc[date, 'lower_ci'] = all_sarima_forecast.loc[closest_date, 'lower_ci']
        sarima_forecast.loc[date, 'upper_ci'] = all_sarima_forecast.loc[closest_date, 'upper_ci']
    
    # Generate Prophet forecast
    # Create future dataframe extending to December 2024
    last_forecast_date = pd.Timestamp("2024-12-31")
    months_to_forecast = (last_forecast_date.year - last_date.year) * 12 + \
                         (last_forecast_date.month - last_date.month) + 1
    
    # Add a buffer to ensure coverage
    months_to_forecast += 3
    
    future = prophet_model.make_future_dataframe(periods=months_to_forecast, freq='M')
    all_prophet_forecast = prophet_model.predict(future)
    
    # Extract only forecasts for our desired period
    prophet_forecast = pd.DataFrame()
    for date in forecast_dates:
        if date <= all_prophet_forecast['ds'].max():
            # Find the closest date in the Prophet forecast
            closest_idx = abs(all_prophet_forecast['ds'] - date).idxmin()
            # Add this row to our forecast
            prophet_forecast = pd.concat([prophet_forecast, all_prophet_forecast.iloc[[closest_idx]]])
        else:
            # For dates beyond the Prophet forecast range, extrapolate using the last available prediction
            last_forecast = all_prophet_forecast.iloc[[-1]].copy()
            last_forecast['ds'] = date
            last_forecast['date'] = date
            prophet_forecast = pd.concat([prophet_forecast, last_forecast])
    
    # Add 'date' column to match the format expected by create_comparison_plot
    prophet_forecast['date'] = prophet_forecast['ds']
    
    return sarima_forecast, prophet_forecast

def main():
    """Main function to run the comparison"""
    # Define paths
    data_path = 'eda_results/unfiltered/data/monthly_aggregated_payments.csv'
    output_dir = 'models/comparison'
    linear_forecast_path = '/Users/Bogodist/work/CupIT_Final/linear_models_forecast.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df, prophet_df = load_data(data_path)
    
    # Generate forecasts from the trained models
    sarima_forecast, prophet_forecast = generate_forecasts_from_models(df, prophet_df)
    
    # Load linear models forecast
    linear_forecast = load_linear_forecast(linear_forecast_path)
    
    # Load saved metrics
    sarima_metrics_text, prophet_metrics_text, sarima_info, prophet_info = load_saved_metrics()
    
    # Create combined comparison plot
    forecast_fig = create_comparison_plot(df, sarima_forecast, prophet_forecast, linear_forecast, output_dir, months_to_show=12)
    
    # Create individual model plots
    sarima_fig = create_individual_model_plot(df, sarima_forecast, 'SARIMA', COLORS['green'], output_dir)
    prophet_fig = create_individual_model_plot(df, prophet_forecast, 'Prophet', COLORS['cyan'], output_dir)
    linear_fig = create_individual_model_plot(df, linear_forecast, 'Linear', COLORS['yellow'], output_dir)
    
    # Create metrics comparison plot
    metrics_fig = create_metrics_comparison_plot(sarima_info, prophet_info, output_dir)
    
    # Calculate summary statistics
    sarima_avg = sarima_forecast['forecast'].mean() / 1e6
    prophet_avg = prophet_forecast['yhat'].mean() / 1e6
    diff_abs = abs(sarima_avg - prophet_avg)
    diff_pct = diff_abs / sarima_avg * 100
    
    # Calculate linear models average if available
    linear_avg = linear_forecast['forecast'].mean() / 1e6 if linear_forecast is not None else 0
    
    # Save summary to file
    with open(f"{output_dir}/forecast_summary.txt", 'w') as f:
        f.write("FORECAST SUMMARY FOR 2024\n")
        f.write("=========================\n\n")
        f.write(f"SARIMA Average Forecast: {sarima_avg:.2f} million ₽\n")
        f.write(f"Prophet Average Forecast: {prophet_avg:.2f} million ₽\n")
        if linear_forecast is not None:
            f.write(f"Linear Models Average Forecast: {linear_avg:.2f} million ₽\n")
        f.write(f"Absolute Difference (SARIMA vs Prophet): {diff_abs:.2f} million ₽\n")
        f.write(f"Percentage Difference: {diff_pct:.2f}%\n\n")
        
        f.write("Monthly Comparison:\n")
        f.write("------------------\n")
        
        # Create comparison dataframe
        comparison_dict = {
            'Month': sarima_forecast.index.strftime('%Y-%m'),
            'SARIMA (million ₽)': sarima_forecast['forecast'] / 1e6,
            'Prophet (million ₽)': prophet_forecast['yhat'].values / 1e6,
        }
        
        # Add linear models if available
        if linear_forecast is not None:
            # Align linear forecast dates with SARIMA dates
            linear_dict = dict(zip(linear_forecast['date'].dt.strftime('%Y-%m'), 
                                  linear_forecast['forecast'] / 1e6))
            linear_values = [linear_dict.get(month, 0) for month in comparison_dict['Month']]
            comparison_dict['Linear (million ₽)'] = linear_values
        
        comparison_df = pd.DataFrame(comparison_dict)
        
        # Add difference columns
        comparison_df['SARIMA-Prophet (million ₽)'] = comparison_df['SARIMA (million ₽)'] - comparison_df['Prophet (million ₽)']
        comparison_df['Diff (%)'] = abs(comparison_df['SARIMA-Prophet (million ₽)']) / comparison_df['SARIMA (million ₽)'] * 100
        
        if linear_forecast is not None:
            comparison_df['SARIMA-Linear (million ₽)'] = comparison_df['SARIMA (million ₽)'] - comparison_df['Linear (million ₽)']
            comparison_df['Diff Linear (%)'] = abs(comparison_df['SARIMA-Linear (million ₽)']) / comparison_df['SARIMA (million ₽)'] * 100
        
        f.write(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main() 