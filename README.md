# covid19-data-tracker

COVID-19 Global Data Tracker
Project Overview
This project provides a comprehensive data analysis tool for tracking and visualizing global COVID-19 trends. It analyzes cases, deaths, recoveries, and vaccinations across countries and time periods using the Our World in Data COVID-19 dataset.
Features

Data cleaning and preprocessing of COVID-19 statistics
Time series analysis of cases, deaths, and vaccinations
Country/region comparison tools
Interactive visualizations including:

Line charts for trend analysis
Bar charts for country comparisons
Heatmaps for correlation analysis
Choropleth maps for global distribution visualization


Customizable reporting options

Prerequisites

Python 3.7+
Required Python libraries:

pandas
numpy
matplotlib
seaborn
plotly
geopandas (optional, for advanced mapping)



Installation

Clone this repository or download the script:
git clone https://github.com/yourusername/covid19-tracker.git

Install the required Python packages:
pip install pandas numpy matplotlib seaborn plotly

For advanced mapping features (optional):
pip install geopandas


Data Source
This project uses the Our World in Data COVID-19 dataset, which is regularly updated and provides comprehensive global COVID-19 statistics.

Download the dataset:

Visit Our World in Data COVID-19 Dataset
Or download directly from their GitHub: OWID GitHub Repository


Save the owid-covid-data.csv file in the project directory.

Usage
Running the Analysis Script
python covid19_tracker.py
Command-line Arguments

--countries: Comma-separated list of countries to analyze (default: World,United States,India,Brazil,United Kingdom)
--start-date: Start date for analysis in YYYY-MM-DD format (default: 2020-01-01)
--end-date: End date for analysis in YYYY-MM-DD format (default: latest available)
--output: Output folder for saving visualizations (default: ./output)
--report: Generate a comprehensive PDF report (requires additional dependencies)

Example:
python covid19_tracker.py --countries "Kenya,South Africa,Nigeria,Egypt" --start-date 2021-01-01 --output ./africa_report
Output
The script generates:

A series of visualization files in the specified output directory
A log file with statistical insights
Optional: A comprehensive PDF report with all visualizations and analysis narratives

Project Structure
covid19-tracker/
├── covid19_tracker.py      # Main analysis script
├── README.md               # This documentation file
├── requirements.txt        # Python dependencies
├── owid-covid-data.csv     # Dataset (needs to be downloaded separately)
└── output/                 # Generated visualizations and reports
    ├── cases_time_series.png
    ├── deaths_time_series.png
    ├── vaccination_progress.png
    ├── country_comparison.png
    ├── world_map.html      # Interactive choropleth map
    └── covid19_report.pdf  # Optional comprehensive report
Analysis Components
The script performs the following key analyses:

Data Cleaning & Preparation

Handling missing values
Date formatting
Country filtering


Time Series Analysis

Daily and cumulative case trends
Moving averages to smooth data
Growth rate calculations


Country Comparisons

Per capita normalizations
Ranking and benchmarking


Vaccination Analysis

Vaccination rollout speed
Population coverage percentages
Dose administration tracking


Geospatial Analysis

Geographic case distribution
Regional hotspot identification



Extending the Project
You can extend this project by:

Adding new visualization types
Incorporating additional data sources for correlation analysis
Implementing predictive modeling components
Creating an interactive dashboard with Dash or Streamlit

Troubleshooting

Missing data errors: Some countries may have incomplete data. Use the --skip-incomplete flag to ignore these.
Memory issues: For large date ranges, use the --sample flag to analyze every nth day.
Visualization errors: Ensure you have the latest versions of matplotlib and seaborn.

Our World in Data for providing the comprehensive COVID-19 dataset
The global scientific community for their tireless efforts during the pandemic
