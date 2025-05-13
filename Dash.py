import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# Load data
df = pd.read_csv("owid-covid-data.csv")
df["date"] = pd.to_datetime(df["date"])

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("COVID-19 Hospitalization & ICU Trends"),
    
    # Dropdown for Country Selection
    dcc.Dropdown(
        id="country-dropdown",
        options=[{"label": country, "value": country} for country in df["location"].unique()],
        value="Kenya"
    ),

    # Date Picker
    dcc.DatePickerRange(
        id="date-range",
        start_date=df["date"].min(),
        end_date=df["date"].max()
    ),

    # Graph Output
    dcc.Graph(id="hospitalization-trend")
])

# Callback for updating graph
@app.callback(
    Output("hospitalization-trend", "figure"),
    [Input("country-dropdown", "value"), Input("date-range", "start_date"), Input("date-range", "end_date")]
)
def update_graph(selected_country, start_date, end_date):
    filtered_df = df[(df["location"] == selected_country) & (df["date"].between(start_date, end_date))]
    if filtered_df.empty:
        fig = px.line(title=f"No data available for {selected_country} in the selected date range")
        fig.update_layout(xaxis_title="Date", yaxis_title="Count")
    else:
        fig = px.line(filtered_df, x="date", y=["hosp_patients", "icu_patients"], title=f"Hospitalization & ICU Trends in {selected_country}")
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Patients", legend_title_text="Patient Type")
    return fig

# Run Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
