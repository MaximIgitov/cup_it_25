#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consistent styling utilities for EDA visualizations
Provides functions to set up professional, presentation-quality visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.io as pio
from plotly.graph_objects import Layout
import numpy as np
import os
from matplotlib import rcParams

# Define color palettes
COLORS = {
    # Base colors
    'blue_dark': '#1a5276',
    'blue': '#2e86c1',
    'cyan': '#3498db',
    'green': '#28b463',
    'lime': '#7dcea0',
    'yellow': '#f4d03f',
    'orange': '#f39c12',
    'red': '#e74c3c',
    
    # Gradient blues for sequential charts
    'blue_grad_0': '#1a5276',
    'blue_grad_1': '#1b5e8a',
    'blue_grad_2': '#1c699e',
    'blue_grad_3': '#1e75b3',
    'blue_grad_4': '#2080c7',
    'blue_grad_5': '#2286c9',
    'blue_grad_6': '#2e8dd0',
    'blue_grad_7': '#3a94d6',
    'blue_grad_8': '#469bdc',
    'blue_grad_9': '#52a2e2',
    'blue_grad_10': '#5ea9e8',
    'blue_grad_11': '#6ab0ee',
    'blue_grad_12': '#76b7f4',
    'blue_grad_13': '#82befa',
    'blue_grad_14': '#8ec5ff',
    
    # Green gradient
    'green_grad_0': '#00441b',
    'green_grad_1': '#006d2c',
    'green_grad_2': '#238b45',
    'green_grad_3': '#41ab5d',
    'green_grad_4': '#66c2a4',
    'green_grad_5': '#99d8c9',
    'green_grad_6': '#ccece6',
    'green_grad_7': '#e5f5f9',
    'green_grad_8': '#f7fcfd',
    'green_grad_9': '#ffffff',
}

def setup_matplotlib_styles():
    """Set up Matplotlib with improved styles for professional visualizations"""
    # Download and use Roboto font if available, otherwise fall back to sans-serif
    try:
        # Try to find Roboto in the system
        roboto_regular = fm.findfont(fm.FontProperties(family='Roboto'))
        roboto_bold = fm.findfont(fm.FontProperties(family='Roboto', weight='bold'))
        
        if 'Roboto' in roboto_regular:
            plt.rcParams['font.family'] = 'Roboto'
        else:
            # If Roboto is not found, use the standard sans-serif
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    except:
        # Fallback to standard sans-serif
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Increase default font sizes for better readability
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 24
    
    # Use better visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Customize the grid for better readability
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Improved tick parameters
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1
    
    # Set figure DPI for sharper images
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Better line styling
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10
    
    # Color palette
    sns.set_palette(list(COLORS.values()))

def setup_plotly_styles():
    """Set up Plotly with improved styles for professional visualizations"""
    # Define the template with Roboto font
    template = {
        "layout": {
            "font": {"family": "Roboto, Arial, sans-serif", "size": 14},
            "title": {
                "font": {"family": "Roboto, Arial, sans-serif", "size": 24}
            },
            "xaxis": {
                "title": {"font": {"family": "Roboto, Arial, sans-serif", "size": 16}},
                "tickfont": {"family": "Roboto, Arial, sans-serif", "size": 14},
                "gridwidth": 1,
                "gridcolor": "rgba(220,220,220,0.5)"
            },
            "yaxis": {
                "title": {"font": {"family": "Roboto, Arial, sans-serif", "size": 16}},
                "tickfont": {"family": "Roboto, Arial, sans-serif", "size": 14},
                "gridwidth": 1,
                "gridcolor": "rgba(220,220,220,0.5)"
            },
            "legend": {
                "font": {"family": "Roboto, Arial, sans-serif", "size": 14}
            },
            "colorway": list(COLORS.values()),
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "margin": {"t": 100, "b": 80, "l": 80, "r": 40},
        }
    }
    
    # Register the template
    pio.templates["custom_template"] = template
    pio.templates.default = "custom_template"

def apply_styles():
    """Apply both Matplotlib and Plotly styles"""
    setup_matplotlib_styles()
    setup_plotly_styles()
    print("Applied professional visualization styles with Roboto font")

def create_custom_figure(figsize=(14, 8)):
    """Create a matplotlib figure with professional styling"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # More subtle grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Thicker axes lines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    return fig, ax

def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """Save a figure with consistent settings"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to {filename}")

def format_currency(x, pos):
    """Format numbers as currency with ruble text instead of symbol"""
    return f"{x:,.0f} руб."

def format_thousands(x, pos):
    """Format y-axis ticks with thousands separator"""
    return f'{int(x):,}'

def format_percent(x, pos):
    """Format y-axis ticks as percentages"""
    return f'{x:.1f}%'

def value_label_bars(ax, fontsize=12, fmt='{:,.0f}', spacing=0.03, color='black'):
    """Add value labels on top of bar charts"""
    for rect in ax.patches:
        height = rect.get_height()
        if height != 0:  # Only label non-zero bars
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + (height * spacing),
                fmt.format(height),
                ha='center',
                va='bottom',
                fontsize=fontsize,
                fontweight='bold',
                color=color
            )

def apply_common_styles(ax, title, xlabel, ylabel):
    """Apply common styling to an axis"""
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.grid(True, alpha=0.3, linestyle='--')

# Apply styles automatically when the module is imported
apply_styles() 