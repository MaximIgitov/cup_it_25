# Medical Services Payment Data Analysis - Unfiltered Approach

This repository contains scripts for comprehensive exploratory data analysis (EDA) of medical services payment data, focusing on temporal patterns, service types, patient demographics, and future payment forecasting. The analysis uses an **unfiltered** approach that preserves all data points.

## Project Structure

The project is organized into the following directories:

```
├── data/
│   ├── raw/             # Raw data files
│   └── unfiltered/      # Unfiltered processed data
├── eda_scripts/         # Analysis scripts
├── eda_results/         # Results of the analysis
│   ├── data/            # Generated data files
│   ├── figures/         # Generated visualizations
│   ├── reports/         # Analysis reports
│   └── unfiltered/      # Unfiltered analysis results
└── run_all_analysis.py  # Main script to run all analyses
```

## Analysis Approach

This repository uses an **unfiltered approach** for data analysis, meaning:
- No data points are removed during preprocessing
- Outliers are preserved in the analysis
- All records are kept for the most comprehensive view of the data

## Analysis Components

The analysis is divided into several components:

1. **Basic Data Processing** (`eda_scripts/eda_main.py`)
   - Data loading with no filtering
   - Basic data cleaning and type conversion
   - Initial exploratory analysis

2. **Temporal Analysis** (`eda_scripts/temporal_analysis.py`)
   - Monthly and yearly payment trends from unfiltered data
   - Seasonal patterns
   - Year-over-year growth analysis

3. **Service Analysis** (`eda_scripts/service_analysis.py`)
   - Distribution of services by type in unfiltered data
   - Service utilization patterns
   - Cost analysis by service type

4. **Patient Analysis** (`eda_scripts/patient_analysis.py`)
   - Patient visit patterns
   - Patient retention analysis
   - Service mix by patient demographics

5. **Forecasting Analysis** (`eda_scripts/forecasting_analysis.py`)
   - Time series decomposition
   - Payment forecasting using SARIMA models
   - Growth rate analysis and projections

## How to Run

To run the complete analysis:

```bash
python run_all_analysis.py
```

To run individual analysis components:

```bash
python eda_scripts/eda_main.py
python eda_scripts/temporal_analysis.py
python eda_scripts/service_analysis.py
python eda_scripts/patient_analysis.py
python eda_scripts/forecasting_analysis.py
```

## Forecasting Results

The forecasting analysis provides:

1. **Seasonal Decomposition**: Breaks down the time series into trend, seasonal, and residual components
2. **Future Payment Projections**: Forecasts payment amounts for the next 6 months with confidence intervals
3. **Growth Rate Analysis**: Detailed analysis of month-over-month and year-over-year growth rates

## Generated Visualizations

The analysis generates various visualizations, including:

- Monthly payment trends
- Year-over-year payment comparisons
- Time series decomposition charts
- Payment forecasts with confidence intervals
- Growth rate analysis charts
- Service mix distributions
- Patient retention analysis

## Data Requirements

The analysis expects a CSV file with the following columns:
- service_document_id
- account_document_id
- service_date
- service_amount_net
- patient_id
- service_code
- service_name

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels

## Dataset Description

The dataset (`CupIT_Sber_data.csv`) contains information about medical services provided to patients, with the following fields:

| Field                | Description                                                    |
|----------------------|----------------------------------------------------------------|
| service_document_id  | Unique identifier for the service document (encoded)           |
| account_document_id  | Unique identifier for the payment document (encoded)           |
| service_date         | Date the service was provided                                  |
| service_amount_net   | Payment amount for the service(s)                              |
| patient_id           | Unique identifier for the patient's medical record (encoded)   |
| service_code         | Service code (encoded)                                         |
| service_name         | Service name (encoded)                                         |
| is_hospital          | Whether the patient was in hospital (0 = no, 1 = yes)          |

## Project Goal

The primary goal of this analysis is to develop models for forecasting monthly aggregated payments, identifying patterns and factors that influence payment volumes over time, using the complete, unfiltered dataset to ensure all data points are considered. 