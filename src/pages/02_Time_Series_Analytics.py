import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from preprocessing.preprocessing_utils import extract_datetime_features
from datetime import datetime, timedelta

st.set_page_config(page_title="Time Series Analytics", layout="wide")

def load_data():
    """Load and preprocess the car rental data"""
    data_path = os.path.join('data', 'processed_car_rental.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['pickup_date'])
        df['dropoff_date'] = pd.to_datetime(df['dropoff_date'])
        return df
    return None

def calculate_rental_duration(row):
    """Calculate rental duration in days"""
    pickup = pd.to_datetime(row['pickup_date'])
    dropoff = pd.to_datetime(row['dropoff_date'])
    return (dropoff - pickup).days

def analyze_time_series(df):
    """Perform time series analysis on the dataset"""
    # Add time-based features
    df_time = extract_datetime_features(df.copy(), 'pickup_date')
    df_time['rental_duration'] = df.apply(calculate_rental_duration, axis=1)
    
    # 1. Rental Volume Trends
    st.subheader("1. Rental Volume Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily rental volume
        daily_rentals = df_time.groupby(pd.to_datetime(df_time['pickup_date']).dt.date).size().reset_index()
        daily_rentals.columns = ['date', 'rentals']
        
        fig_daily = px.line(daily_rentals, x='date', y='rentals',
                          title='Daily Rental Volume',
                          labels={'date': 'Date', 'rentals': 'Number of Rentals'})
        fig_daily.update_layout(showlegend=True)
        st.plotly_chart(fig_daily)
    
    with col2:
        # Weekly rental volume
        weekly_rentals = df_time.groupby(pd.to_datetime(df_time['pickup_date']).dt.isocalendar().week).size().reset_index()
        weekly_rentals.columns = ['week', 'rentals']
        
        fig_weekly = px.line(weekly_rentals, x='week', y='rentals',
                           title='Weekly Rental Volume',
                           labels={'week': 'Week Number', 'rentals': 'Number of Rentals'})
        st.plotly_chart(fig_weekly)

    # 2. Peak Hours Heatmap
    st.subheader("2. Peak Hours Analysis")
    
    # Create hour-day heatmap
    hourly_day_rentals = df_time.groupby(['pickup_date_dayofweek', 'pickup_date_hour']).size().reset_index()
    hourly_day_rentals.columns = ['day', 'hour', 'rentals']
    hourly_day_rentals['day'] = hourly_day_rentals['day'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    fig_heatmap = px.density_heatmap(hourly_day_rentals, x='hour', y='day', z='rentals',
                                    title='Rental Activity Heatmap by Hour and Day',
                                    labels={'hour': 'Hour of Day', 'day': 'Day of Week', 'rentals': 'Number of Rentals'})
    st.plotly_chart(fig_heatmap)

    # 3. Rental Duration Analysis
    st.subheader("3. Rental Duration Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution
        fig_duration = px.histogram(df_time, x='rental_duration',
                                  title='Distribution of Rental Durations',
                                  labels={'rental_duration': 'Duration (Days)', 'count': 'Number of Rentals'},
                                  nbins=30)
        st.plotly_chart(fig_duration)
    
    with col2:
        # Average duration by vehicle type
        avg_duration = df_time.groupby('vehicle.type')['rental_duration'].mean().reset_index()
        fig_avg_duration = px.bar(avg_duration, x='vehicle.type', y='rental_duration',
                                 title='Average Rental Duration by Vehicle Type',
                                 labels={'vehicle.type': 'Vehicle Type', 'rental_duration': 'Average Duration (Days)'})
        st.plotly_chart(fig_avg_duration)

    # 4. Seasonal Patterns
    st.subheader("4. Seasonal Patterns")
    
    # Monthly trends by vehicle type
    monthly_type_rentals = df_time.groupby(['pickup_date_month', 'vehicle.type']).size().reset_index()
    monthly_type_rentals.columns = ['month', 'vehicle_type', 'rentals']
    
    fig_seasonal = px.line(monthly_type_rentals, x='month', y='rentals', color='vehicle_type',
                          title='Monthly Rental Trends by Vehicle Type',
                          labels={'month': 'Month', 'rentals': 'Number of Rentals', 'vehicle_type': 'Vehicle Type'})
    st.plotly_chart(fig_seasonal)

    # 5. Rate Analysis Over Time
    st.subheader("5. Rate Analysis")
    
    # Daily rate trends
    df_time['date'] = pd.to_datetime(df_time['pickup_date']).dt.date
    daily_rates = df_time.groupby('date')['rate.daily'].mean().reset_index()
    
    fig_rates = px.line(daily_rates, x='date', y='rate.daily',
                        title='Average Daily Rate Trends',
                        labels={'date': 'Date', 'rate.daily': 'Average Daily Rate ($)'})
    st.plotly_chart(fig_rates)

def main():
    st.title("Time Series Analytics Dashboard")
    
    df = load_data()
    
    if df is not None:
        # Display data info
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Date range: {pd.to_datetime(df['pickup_date']).min().date()} to {pd.to_datetime(df['pickup_date']).max().date()}")
        with col2:
            st.write("Columns with datetime data:")
            st.write(["pickup_date", "dropoff_date"])
        
        # Perform time series analysis
        analyze_time_series(df)
    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
