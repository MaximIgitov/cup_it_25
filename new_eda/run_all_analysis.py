#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run all unfiltered EDA analyses in sequence
This script coordinates the execution of all analytical modules with the unfiltered approach
"""

import os
import time
import subprocess
import sys

def run_script(script_path, description):
    """Run a Python script and measure execution time"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Run the script as a subprocess
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_path}. Return code: {result.returncode}")
        return False
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n✅ Completed: {description} in {elapsed:.2f} seconds")
    return True

def main():
    """Run all unfiltered EDA scripts in the appropriate order"""
    # Create required directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/unfiltered', exist_ok=True)
    os.makedirs('eda_results/unfiltered/data', exist_ok=True)
    os.makedirs('eda_results/unfiltered/figures', exist_ok=True)
    os.makedirs('eda_results/unfiltered/reports', exist_ok=True)
    
    # List of scripts to run with descriptions
    scripts = [
        ("eda_scripts/eda_main.py", "Unfiltered Data Loading and Preprocessing"),
        ("eda_scripts/temporal_analysis.py", "Unfiltered Temporal Analysis"),
        ("eda_scripts/service_analysis.py", "Unfiltered Service Analysis"),
        ("eda_scripts/patient_analysis.py", "Unfiltered Patient Analysis"),
        ("eda_scripts/forecasting_analysis.py", "Unfiltered Forecasting Analysis")
    ]
    
    # Run all scripts in sequence
    all_successful = True
    
    for script_path, description in scripts:
        if not run_script(script_path, description):
            all_successful = False
            print(f"\nWarning: Skipping remaining scripts due to error in {script_path}")
            break
    
    if all_successful:
        print("\n🎉 All unfiltered analyses completed successfully!")
    else:
        print("\n⚠️ Some analyses did not complete successfully. Please check the logs above.")

if __name__ == "__main__":
    print("\n🚀 Starting comprehensive unfiltered EDA analysis...")
    main() 