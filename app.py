import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("owid-covid-data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.title("Filter Data")
selected_country = st.sidebar.selectbox("Select Country", df["location"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

# Validate date range input
if len(date_range) != 2 or date_range[0] > date_range[1]:
    st.sidebar.error("Please select a valid date range where the start date is before the end date.")
else:
    # Filter data based on user input
    filtered_df = df[(df["location"] == selected_country) & (df["date"].between(date_range[0], date_range[1]))]

    # Plot ICU and Hospitalization Data
    fig = px.line(filtered_df, x="date", y=["hosp_patients", "icu_patients"], title=f"Hospitalization & ICU Trends in {selected_country}")
    st.plotly_chart(fig)

# Main page title and description
st.title("COVID-19 Hospitalization & ICU Trends")
st.write("This app visualizes hospitalization and ICU patient trends for COVID-19 by country and date range.")
