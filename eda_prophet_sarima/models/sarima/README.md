# SARIMA Model for Monthly Aggregated Payments

This directory contains a complete implementation of a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for forecasting monthly aggregated payments.

## Features

- SARIMA model implementation with hyperparameter tuning
- Forecasting of payments for the next 12 months
- Statistical tests for time series analysis
- Professional visualization of historical data and forecasts
- Comprehensive evaluation metrics

## Files

- `train_sarima_model.py`: Main script for training the SARIMA model
- `model_info.json`: Model parameters and evaluation metrics
- `hyperparameter_tuning_results.csv`: Results of hyperparameter tuning
- `forecast.csv`: Forecasted values for the next 12 months
- `model_summary.txt`: Summary of the SARIMA model
- `sarima_model.pkl`: Serialized SARIMA model
- `plots/`: Directory containing visualizations
  - `original_time_series.png`: Plot of the original time series
  - `acf_pacf.png`: Autocorrelation and partial autocorrelation plots
  - `forecast.png`: Plot of historical data with forecast
  - `residuals.png`: Plot of model residuals
  - `residuals_distribution.png`: Distribution of model residuals

## Usage

To train the SARIMA model and generate forecasts, run:

```bash
python models/sarima/train_sarima_model.py
```

## Methodology

1. **Data Loading**: The script loads monthly aggregated payments data from 2022-2023
2. **Data Splitting**: The data is split into training (first 21 months) and test (last 3 months) sets
3. **Statistical Tests**: Performs ADF, KPSS, and Ljung-Box tests to analyze time series properties
4. **Hyperparameter Tuning**: Conducts grid search over a wide range of SARIMA parameters
   - **Model Selection**: Selects the best model based on forecasting performance (MAPE) on the test set
   - This ensures the model is optimized for its actual forecasting task rather than in-sample fit
5. **Model Training**: Trains the SARIMA model using the best parameters
6. **Evaluation**: Evaluates the model on the test set (last 3 months of historical data)
7. **Forecasting**: Generates forecasts for the next 12 months with confidence intervals
8. **Visualization**: Creates professional plots of the time series and forecasts

## Statistical Tests

The script performs the following statistical tests:

- **Augmented Dickey-Fuller (ADF) Test**: Tests for stationarity
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**: Alternative test for stationarity
- **Ljung-Box Test**: Tests for autocorrelation in the time series

## Evaluation Metrics

The model is evaluated using:

- **Mean Absolute Error (MAE)**: Average absolute difference between observed and predicted values
- **Root Mean Squared Error (RMSE)**: Square root of the average squared differences
- **Mean Absolute Percentage Error (MAPE)**: Average percentage difference between observed and predicted values 