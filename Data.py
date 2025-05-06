# Python for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('owid-covid-data.csv')


# Check the columns in the dataset
print("Columns in dataset:", df.columns)

# Preview the first few rows
print("\nDataset preview:")
print(df.head())

# Identify missing values in the dataset
print("\nMissing values count:")
print(df.isnull().sum())




# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Filter for selected countries (Kenya, USA, India)
selected_countries = ['Kenya', 'United States', 'India']
df = df[df['location'].isin(selected_countries)]

# Drop rows where critical values (date, total_cases, total_deaths) are missing
df = df.dropna(subset=['date', 'total_cases', 'total_deaths'])

# Handle missing numeric values (fill with interpolation)
df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
df.fillna(method='bfill', inplace=True)  # Back-fill as needed

# Display the cleaned dataset info
print("\nCleaned Dataset Info:")
print(df.info())

# Preview cleaned data
print("\nCleaned Data Preview:")
print(df.head())


# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Function to plot total cases over time for selected countries
def plot_total_cases(countries):
    plt.figure(figsize=(15, 8))
    for country in countries:
        country_data = df[df['location'] == country]
        plt.plot(country_data['date'], country_data['total_cases'], label=country)
    plt.title('Total COVID-19 Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to calculate the death rate (total_deaths/total_cases)
def calculate_death_rate(df):
    df['death_rate'] = df['total_deaths'] / df['total_cases'].replace(0, np.nan) * 100
    return df

# Generate descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Display the first few rows of the dataset
print("\nDataset Preview:")
print(df.head())

# #bar chart
df['continent'].value_counts().plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Number of Countries per Continent')
plt.xlabel('Continent')
plt.ylabel('Number of Countries')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Aggregate total deaths by country
deaths_by_country = df.groupby('location')['total_deaths'].max()

# Select top countries for visualization (adjust as needed)
top_countries = deaths_by_country.nlargest(5)  # Selects the top 5 countries with highest deaths

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_countries, labels=top_countries.index, autopct='%1.1f%%', startangle=140, colors=['red', 'orange', 'yellow', 'purple', 'blue'])
plt.title('COVID-19 Total Deaths Distribution Among Countries')
plt.show()


def plot_vaccination_trends(countries):
    plt.figure(figsize=(15, 8))
    for country in countries:
        country_data = df[df['location'] == country]
        plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
    plt.title('Cumulative COVID-19 Vaccinations Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Vaccinations')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
