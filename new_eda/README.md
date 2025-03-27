# Medical Services Payment Data Analysis

## Overview

This project focuses on the analysis of medical services payment data, providing insights into payment trends, service types, patient demographics, and forecasting. The analysis is performed using an unfiltered approach that preserves all data points without removing outliers or duplicates.

## Project Structure

```
.
├── data/
│   ├── raw/             # Raw data files
│   └── unfiltered/      # Unfiltered processed data
├── eda_scripts/         # Analysis scripts
│   ├── eda_main.py       # Data loading and preprocessing
│   ├── temporal_analysis.py  # Temporal analysis of payments
│   ├── service_analysis.py   # Service type analysis
│   ├── patient_analysis.py   # Patient demographics and behavior
│   └── forecasting_analysis.py  # Time series forecasting
├── eda_results/         # Results of analyses
│   └── unfiltered/
│       ├── data/        # Generated data files
│       └── figures/     # Generated visualizations
│           ├── temporal/     # Temporal analysis figures
│           ├── service/      # Service analysis figures
│           ├── patient/      # Patient analysis figures
│           └── forecasting/  # Forecasting figures
└── run_all_analysis.py  # Main script to run all analyses
```

## Analysis Components

The project consists of several key analysis components:

1. **Data Preprocessing** (`eda_main.py`)
   - Loads raw data and performs minimal preprocessing
   - Preserves all data points without filtering
   - Saves processed data to `data/unfiltered/`

2. **Temporal Analysis** (`temporal_analysis.py`)
   - Analyzes monthly and yearly payment trends
   - Calculates year-over-year growth rates
   - Visualizes payment comparisons between years

3. **Service Analysis** (`service_analysis.py`)
   - Identifies top service types by payment volume
   - Analyzes service type trends over time
   - Calculates growth rates for different service types
   - Examines average payment per service

4. **Patient Analysis** (`patient_analysis.py`)
   - Analyzes patient visit patterns
   - Calculates patient retention rates
   - Segments patients based on visit frequency
   - Examines service mix by patient segment

5. **Forecasting Analysis** (`forecasting_analysis.py`)
   - Decomposes time series data to identify trends and seasonality
   - Forecasts future payments using SARIMA models
   - Analyzes growth rates and seasonality patterns

## Requirements

This project requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- scipy

To install the required packages, run:
```
pip install -r requirements.txt
```

## Running the Analyses

You can run all analyses at once using the main script:

```
python run_all_analysis.py
```

Or run each analysis script individually:

```
python eda_scripts/eda_main.py              # Run data preprocessing
python eda_scripts/temporal_analysis.py      # Run temporal analysis
python eda_scripts/service_analysis.py       # Run service analysis
python eda_scripts/patient_analysis.py       # Run patient analysis
python eda_scripts/forecasting_analysis.py   # Run forecasting analysis
```

## Results

Analysis results are organized into subdirectories:

- `eda_results/unfiltered/data/`: CSV files with analysis results
- `eda_results/unfiltered/figures/`: Visualizations from each analysis:
  - `temporal/`: Year-over-year comparisons and growth charts
  - `service/`: Service type distributions and trends
  - `patient/`: Patient demographics and behavior visualizations
  - `forecasting/`: Time series decomposition and forecast charts

## Approach

This project uses an unfiltered approach that preserves all data points without removing outliers or duplicates. This approach provides a comprehensive view of the data but may be sensitive to data quality issues. The analysis scripts are designed to be modular and can be extended or modified as needed.

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