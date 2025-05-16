#!/usr/bin/env python3
"""
COVID-19 Global Data Tracker and Analysis Tool
==============================================

This script analyzes the Our World in Data COVID-19 dataset to track global
COVID-19 trends including cases, deaths, and vaccinations across countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import plotly for maps (but continue if not available)
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    # Enable offline mode for plotly
    pio.renderers.default = "browser"
except ImportError:
    print("Plotly not installed. Choropleth maps will be skipped.")
    PLOTLY_AVAILABLE = False

# Set up visualization styles
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Output directory for saving visualizations
OUTPUT_DIR = "covid19_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_explore_data(file_path='owid-covid-data.csv'):
    """Load and perform initial exploration of the COVID-19 dataset"""
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please download the file from https://github.com/owid/covid-19-data/tree/master/public/data")
        return None
    
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Display basic info
    print("\nColumns in the dataset:")
    print(", ".join(df.columns.tolist()))
    
    # Count unique locations
    location_count = len(df['location'].unique())
    print(f"\nTotal locations in dataset: {location_count}")
    
    return df

def clean_data(df, key_countries=None):
    """Clean and prepare data for analysis"""
    if df is None:
        return None, None
    
    print("\nCleaning data...")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a copy of the cleaned dataset
    df_clean = df.copy()
    
    # Default key countries if none provided
    if key_countries is None:
        key_countries = ['United States', 'India', 'Brazil', 'United Kingdom', 
                         'Russia', 'France', 'Germany', 'Italy', 'China', 
                         'South Africa', 'Kenya']
    
    # Create a filtered dataset with just these countries
    df_key_countries = df_clean[df_clean['location'].isin(key_countries)]
    
    # Handle missing values for cases and deaths
    for col in ['new_cases', 'new_deaths']:
        df_key_countries[col] = df_key_countries[col].fillna(0)

    for col in ['total_cases', 'total_deaths']:
        # Forward fill for cumulative metrics (using previous day's value)
        df_key_countries[col] = df_key_countries.groupby('location')[col].transform(
            lambda x: x.fillna(method='ffill')
        )
    
    # For countries with no reported cases/deaths at beginning, fill with 0
    df_key_countries[['total_cases', 'total_deaths']] = df_key_countries[['total_cases', 'total_deaths']].fillna(0)
    
    print("Data cleaning completed")
    return df_clean, df_key_countries

def global_trends_analysis(df_clean):
    """Analyze and visualize global COVID-19 trends"""
    print("\nAnalyzing global trends...")
    
    # Get world data
    if 'World' in df_clean['location'].unique():
        world_data = df_clean[df_clean['location'] == 'World'].sort_values('date')
    else:
        # If no 'World' aggregate data, create it
        world_data = df_clean.groupby('date')[['new_cases', 'new_deaths', 'total_cases', 'total_deaths']].sum().reset_index()
    
    # Plot global cumulative cases and deaths over time
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(world_data['date'], world_data['total_cases'], 'b-', linewidth=2)
    plt.title('Global Cumulative COVID-19 Cases')
    plt.ylabel('Total Cases')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(world_data['date'], world_data['total_deaths'], 'r-', linewidth=2)
    plt.title('Global Cumulative COVID-19 Deaths')
    plt.xlabel('Date')
    plt.ylabel('Total Deaths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/global_cumulative_trends.png")
    print(f"Saved chart to {OUTPUT_DIR}/global_cumulative_trends.png")
    
    # Plot daily new cases and deaths with 7-day moving average for smoothing
    plt.figure(figsize=(14, 8))
    
    # Calculate 7-day moving averages
    world_data['new_cases_smoothed'] = world_data['new_cases'].rolling(window=7).mean()
    world_data['new_deaths_smoothed'] = world_data['new_deaths'].rolling(window=7).mean()
    
    plt.subplot(2, 1, 1)
    plt.bar(world_data['date'], world_data['new_cases'], color='skyblue', alpha=0.3, label='Daily Cases')
    plt.plot(world_data['date'], world_data['new_cases_smoothed'], color='blue', linewidth=2, label='7-day Average')
    plt.title('Global Daily New COVID-19 Cases')
    plt.ylabel('New Cases')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(world_data['date'], world_data['new_deaths'], color='salmon', alpha=0.3, label='Daily Deaths')
    plt.plot(world_data['date'], world_data['new_deaths_smoothed'], color='red', linewidth=2, label='7-day Average')
    plt.title('Global Daily New COVID-19 Deaths')
    plt.xlabel('Date')
    plt.ylabel('New Deaths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/global_daily_trends.png")
    print(f"Saved chart to {OUTPUT_DIR}/global_daily_trends.png")
    
    # Calculate key statistics
    total_cases = world_data['total_cases'].max()
    total_deaths = world_data['total_deaths'].max()
    peak_cases_day = world_data.loc[world_data['new_cases'].idxmax()]
    peak_deaths_day = world_data.loc[world_data['new_deaths'].idxmax()]
    
    print(f"Total global cases: {total_cases:,.0f}")
    print(f"Total global deaths: {total_deaths:,.0f}")
    print(f"Peak daily cases: {peak_cases_day['new_cases']:,.0f} on {peak_cases_day['date'].strftime('%Y-%m-%d')}")
    print(f"Peak daily deaths: {peak_deaths_day['new_deaths']:,.0f} on {peak_deaths_day['date'].strftime('%Y-%m-%d')}")
    
    return world_data

def country_comparison(df_key_countries):
    """Compare COVID-19 metrics across selected countries"""
    if df_key_countries is None or df_key_countries.empty:
        print("No data available for country comparison")
        return
    
    print("\nComparing metrics across countries...")
    
    # Get the latest data for each country
    latest_date = df_key_countries['date'].max()
    latest_data = df_key_countries[df_key_countries['date'] == latest_date].copy()
    
    # Calculate death rate
    latest_data['death_rate'] = (latest_data['total_deaths'] / latest_data['total_cases'] * 100).round(2)
    
    # Sort countries by total cases
    top_countries_by_cases = latest_data.sort_values('total_cases', ascending=False)
    
    # Plot total cases by country
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_cases', y='location', data=top_countries_by_cases, palette='viridis')
    plt.title('Total COVID-19 Cases by Country')
    plt.xlabel('Total Cases')
    plt.ylabel('Country')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_cases_by_country.png")
    print(f"Saved chart to {OUTPUT_DIR}/total_cases_by_country.png")
    
    # Plot total deaths by country
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_deaths', y='location', data=top_countries_by_cases, palette='rocket')
    plt.title('Total COVID-19 Deaths by Country')
    plt.xlabel('Total Deaths')
    plt.ylabel('Country')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_deaths_by_country.png")
    print(f"Saved chart to {OUTPUT_DIR}/total_deaths_by_country.png")
    
    # Plot death rates
    plt.figure(figsize=(12, 8))
    sns.barplot(x='death_rate', y='location', data=top_countries_by_cases, palette='coolwarm')
    plt.title('COVID-19 Death Rate by Country (%)')
    plt.xlabel('Death Rate (%)')
    plt.ylabel('Country')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/death_rate_by_country.png")
    print(f"Saved chart to {OUTPUT_DIR}/death_rate_by_country.png")
    
    # Print summary
    print("\nSummary of latest data by country:")
    summary_data = top_countries_by_cases[['location', 'total_cases', 'total_deaths', 'death_rate']]
    for _, row in summary_data.iterrows():
        print(f"{row['location']}: {row['total_cases']:,.0f} cases, {row['total_deaths']:,.0f} deaths, {row['death_rate']:.2f}% death rate")
    
    return latest_data

def time_series_analysis(df_key_countries):
    """Analyze time series data for selected countries"""
    if df_key_countries is None or df_key_countries.empty:
        print("No data available for time series analysis")
        return
    
    print("\nPerforming time series analysis...")
    
    # Get list of countries
    countries = df_key_countries['location'].unique()
    
    # Plot total cases over time for key countries
    plt.figure(figsize=(14, 10))
    for country in countries:
        country_data = df_key_countries[df_key_countries['location'] == country]
        plt.plot(country_data['date'], country_data['total_cases'], linewidth=2, label=country)

    plt.title('COVID-19 Total Cases Over Time by Country')
    plt.xlabel('Date')
    plt.ylabel('Total Cases (log scale)')
    plt.yscale('log')  # Log scale to better visualize countries with different case numbers
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_cases_over_time.png")
    print(f"Saved chart to {OUTPUT_DIR}/total_cases_over_time.png")

    # Plot new cases (7-day moving average) for better visualization
    plt.figure(figsize=(14, 10))
    for country in countries:
        country_data = df_key_countries[df_key_countries['location'] == country].copy()
        # Calculate 7-day moving average
        country_data['new_cases_smoothed'] = country_data['new_cases'].rolling(window=7).mean()
        plt.plot(country_data['date'], country_data['new_cases_smoothed'], linewidth=2, label=country)

    plt.title('COVID-19 Daily New Cases (7-day Average) by Country')
    plt.xlabel('Date')
    plt.ylabel('New Cases (7-day avg)')
    plt.yscale('log')  # Log scale to better visualize countries with different case numbers
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/new_cases_over_time.png")
    print(f"Saved chart to {OUTPUT_DIR}/new_cases_over_time.png")
    
    # Plot total deaths over time
    plt.figure(figsize=(14, 10))
    for country in countries:
        country_data = df_key_countries[df_key_countries['location'] == country]
        plt.plot(country_data['date'], country_data['total_deaths'], linewidth=2, label=country)

    plt.title('COVID-19 Total Deaths Over Time by Country')
    plt.xlabel('Date')
    plt.ylabel('Total Deaths (log scale)')
    plt.yscale('log')  # Log scale to better visualize countries with different death numbers
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_deaths_over_time.png")
    print(f"Saved chart to {OUTPUT_DIR}/total_deaths_over_time.png")
    
    print("Time series analysis completed")

def vaccination_analysis(df_clean, df_key_countries):
    """Analyze vaccination data if available"""
    if df_clean is None or df_key_countries is None:
        print("No data available for vaccination analysis")
        return
    
    # Check if vaccination data is available
    vaccination_cols = [col for col in df_clean.columns if 'vaccine' in col.lower()]
    print(f"\nAvailable vaccination columns: {', '.join(vaccination_cols) if vaccination_cols else 'None'}")
    
    if 'total_vaccinations' not in df_clean.columns:
        print("Vaccination data not available in this dataset")
        return
    
    print("\nAnalyzing vaccination data...")
    
    # Filter for key countries and non-null vaccination data
    vax_data = df_key_countries[
        df_key_countries['total_vaccinations'].notnull()
    ].copy()
    
    if vax_data.empty:
        print("No vaccination data available for selected countries")
        return
    
    # Get list of countries with vaccination data
    countries_with_vax = vax_data['location'].unique()
    
    # Plot vaccination progress over time for key countries
    plt.figure(figsize=(14, 10))
    for country in countries_with_vax:
        country_vax = vax_data[vax_data['location'] == country]
        if not country_vax.empty:  # Only plot if we have vaccination data
            plt.plot(country_vax['date'], country_vax['total_vaccinations'], linewidth=2, label=country)
    
    plt.title('COVID-19 Total Vaccinations Over Time by Country')
    plt.xlabel('Date')
    plt.ylabel('Total Vaccinations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_vaccinations_over_time.png")
    print(f"Saved chart to {OUTPUT_DIR}/total_vaccinations_over_time.png")
    
    # Calculate percentage of population vaccinated (if people_vaccinated and population columns exist)
    if 'people_vaccinated' in df_clean.columns and 'population' in df_clean.columns:
        # Get latest vaccination data for each country
        latest_vax_date = vax_data['date'].max()
        latest_vax = vax_data[vax_data['date'] == latest_vax_date].copy()
        
        # Calculate vaccination percentage
        latest_vax['vax_percentage'] = (latest_vax['people_vaccinated'] / latest_vax['population'] * 100).round(2)
        
        # Sort and plot
        latest_vax = latest_vax.sort_values('vax_percentage', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='vax_percentage', y='location', data=latest_vax, palette='viridis')
        plt.title('Percentage of Population Vaccinated by Country')
        plt.xlabel('Population Vaccinated (%)')
        plt.ylabel('Country')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/vaccination_percentage_by_country.png")
        print(f"Saved chart to {OUTPUT_DIR}/vaccination_percentage_by_country.png")
        
        # Print summary
        print("\nLatest vaccination data:")
        for _, row in latest_vax.iterrows():
            print(f"{row['location']}: {row['vax_percentage']:.2f}% of population vaccinated")
        
        return latest_vax
    else:
        print("Population or people_vaccinated data not available for percentage calculations")
        return None

def create_choropleth_maps(df_clean):
    """Create choropleth maps if plotly is available"""
    if df_clean is None:
        print("No data available for choropleth maps")
        return
    
    if not PLOTLY_AVAILABLE:
        print("\nPlotly not available. Skipping choropleth maps.")
        return
    
    print("\nCreating choropleth maps...")
    
    # Check if we have the necessary data for a choropleth map
    if 'iso_code' not in df_clean.columns:
        print("ISO codes not available for choropleth mapping")
        return
    
    # Get latest data for all countries
    latest_global_date = df_clean['date'].max()
    latest_global = df_clean[df_clean['date'] == latest_global_date].copy()
    
    # Cases map
    if 'total_cases_per_million' in latest_global.columns:
        fig1 = px.choropleth(latest_global, 
                            locations="iso_code",
                            color="total_cases_per_million", 
                            hover_name="location",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="COVID-19 Cases per Million People")
        fig1.write_html(f"{OUTPUT_DIR}/cases_per_million_map.html")
        print(f"Saved map to {OUTPUT_DIR}/cases_per_million_map.html")
    
    # Deaths map
    if 'total_deaths_per_million' in latest_global.columns:
        fig2 = px.choropleth(latest_global, 
                            locations="iso_code",
                            color="total_deaths_per_million", 
                            hover_name="location",
                            color_continuous_scale=px.colors.sequential.Reds,
                            title="COVID-19 Deaths per Million People")
        fig2.write_html(f"{OUTPUT_DIR}/deaths_per_million_map.html")
        print(f"Saved map to {OUTPUT_DIR}/deaths_per_million_map.html")
    
    # Vaccination map (if data is available)
    if 'people_vaccinated_per_hundred' in latest_global.columns:
        # Filter for non-null vaccination data
        vax_map_data = latest_global[latest_global['people_vaccinated_per_hundred'].notnull()]
        
        fig3 = px.choropleth(vax_map_data, 
                            locations="iso_code",
                            color="people_vaccinated_per_hundred", 
                            hover_name="location",
                            color_continuous_scale=px.colors.sequential.Greens,
                            title="COVID-19 Vaccination Rate (% of Population)")
        fig3.write_html(f"{OUTPUT_DIR}/vaccination_rate_map.html")
        print(f"Saved map to {OUTPUT_DIR}/vaccination_rate_map.html")

def generate_insights(world_data, latest_country_data, latest_vax_data=None):
    """Generate key insights from the data"""
    if world_data is None or latest_country_data is None:
        print("Insufficient data for insights")
        return
    
    print("\n======= KEY INSIGHTS =======")
    
    # Calculate global stats
    total_global_cases = world_data['total_cases'].max()
    total_global_deaths = world_data['total_deaths'].max()
    global_death_rate = (total_global_deaths / total_global_cases * 100).round(2)
    
    # Find peak periods
    peak_cases_date = world_data.loc[world_data['new_cases'].idxmax()]['date']
    peak_deaths_date = world_data.loc[world_data['new_deaths'].idxmax()]['date']
    
    # Calculate most affected countries
    most_cases_country = latest_country_data.loc[latest_country_data['total_cases'].idxmax()]
    most_deaths_country = latest_country_data.loc[latest_country_data['total_deaths'].idxmax()]
    highest_death_rate_country = latest_country_data.loc[latest_country_data['death_rate'].idxmax()]
    
    print(f"1. GLOBAL IMPACT: The COVID-19 pandemic has resulted in {total_global_cases:,.0f} cases and {total_global_deaths:,.0f} deaths worldwide as of {world_data['date'].max().strftime('%Y-%m-%d')}, with a global death rate of {global_death_rate}%.")
    
    print(f"2. REGIONAL DISPARITIES: {most_cases_country['location']} experienced the highest number of total cases ({most_cases_country['total_cases']:,.0f}), while {highest_death_rate_country['location']} had the highest mortality rate relative to cases ({highest_death_rate_country['death_rate']:.2f}%).")
    
    print(f"3. WAVES AND PATTERNS: The data reveals multiple global waves of infection, with the most significant spike in cases occurring on {peak_cases_date.strftime('%Y-%m-%d')} and deaths on {peak_deaths_date.strftime('%Y-%m-%d')}.")
    
    # Vaccination insights if available
    if latest_vax_data is not None and not latest_vax_data.empty:
        highest_vax_country = latest_vax_data.loc[latest_vax_data['vax_percentage'].idxmax()]
        lowest_vax_country = latest_vax_data.loc[latest_vax_data['vax_percentage'].idxmin()]
        
        print(f"4. VACCINATION PROGRESS: {highest_vax_country['location']} leads vaccination efforts with {highest_vax_country['vax_percentage']:.2f}% of the population vaccinated, while {lowest_vax_country['location']} has vaccinated just {lowest_vax_country['vax_percentage']:.2f}% of its population.")
    
    # Calculate correlation between new cases and deaths with lag
    lag_days = 14  # Typically deaths lag cases by around 2 weeks
    world_data_copy = world_data.copy()
    world_data_copy['new_deaths_lag'] = world_data_copy['new_deaths'].shift(-lag_days)
    correlation = world_data_copy['new_cases'].corr(world_data_copy['new_deaths_lag'])
    
    correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
    print(f"5. CORRELATION FINDINGS: There is a {correlation_strength} correlation (r = {correlation:.2f}) between new cases and deaths 14 days later, confirming the expected lag between infection and mortality.")
    
    print("\n==== LIMITATIONS OF ANALYSIS ====")
    print("- Data reporting inconsistencies across countries")
    print("- Testing availability affecting case counts")
    print("- Different definitions for COVID-19 deaths")
    print("- Vaccination data availability varies by country")
    
    print("\n==== RECOMMENDATIONS ====")
    print("1. Continue monitoring trends as the pandemic evolves")
    print("2. Compare countries with different containment strategies to identify effective measures")
    print("3. Analyze the relationship between vaccination rates and case/death trends")
    print("4. Consider socioeconomic factors in future analyses")
    print("5. Examine healthcare system capacity against peak infection periods")

def main():
    """Main function to run the entire analysis"""
    print("="*50)
    print("COVID-19 GLOBAL DATA TRACKER AND ANALYSIS")
    print("="*50)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Step 2: Clean data
    df_clean, df_key_countries = clean_data(df)
    
    # Step 3: Global trends analysis
    world_data = global_trends_analysis(df_clean)
    
    # Step 4: Country comparison
    latest_country_data = country_comparison(df_key_countries)
    
    # Step 5: Time series analysis
    time_series_analysis(df_key_countries)
    
    # Step 6: Vaccination analysis
    latest_vax_data = vaccination_analysis(df_clean, df_key_countries)
    
    # Step 7: Choropleth maps
    create_choropleth_maps(df_clean)
    
    # Step 8: Generate insights
    generate_insights(world_data, latest_country_data, latest_vax_data)
    
    print("\nAnalysis complete! All results saved to the '{OUTPUT_DIR}' directory.")
    print(f"Check {OUTPUT_DIR} for visualizations and charts.")

if __name__ == "__main__":
    main()
