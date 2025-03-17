import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from preprocessing.preprocessing_utils import extract_datetime_features
import numpy as np

st.set_page_config(page_title="Advanced Business Analytics", layout="wide")

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

def analyze_advanced_metrics(df):
    """Perform advanced business analytics"""
    # Add time-based features
    df_time = extract_datetime_features(df.copy(), 'pickup_date')
    df_time['rental_duration'] = df.apply(calculate_rental_duration, axis=1)
    df_time['revenue'] = df_time['rental_duration'] * df_time['rate.daily']
    
    # 1. Brand Popularity by City Over Time
    st.subheader("1. Brand Market Share Evolution by City")
    
    # Get top cities by rental volume
    top_cities = df_time.groupby('pickUp.city').size().nlargest(10).index.tolist()
    
    # City selector
    selected_cities = st.multiselect(
        "Select cities to analyze (max 6):",
        top_cities,
        default=top_cities[:3]
    )
    
    if len(selected_cities) > 0:
        # Filter for selected cities
        city_data = df_time[df_time['pickUp.city'].isin(selected_cities)]
        
        # Calculate market share per city and brand over time
        brand_city_share = city_data.groupby(['pickup_date_month', 'pickUp.city', 'vehicle.make']).size().reset_index(name='rentals')
        brand_city_share['market_share'] = brand_city_share.groupby(['pickup_date_month', 'pickUp.city'])['rentals'].transform(lambda x: x / x.sum() * 100)
        
        # Get top brands for better visualization
        top_brands = brand_city_share.groupby('vehicle.make')['rentals'].sum().nlargest(5).index
        brand_city_share = brand_city_share[brand_city_share['vehicle.make'].isin(top_brands)]
        
        # Create line plot for each city
        for city in selected_cities:
            city_data = brand_city_share[brand_city_share['pickUp.city'] == city]
            fig_brand_share = px.line(city_data,
                                    x='pickup_date_month',
                                    y='market_share',
                                    color='vehicle.make',
                                    title=f'Brand Market Share Evolution in {city}',
                                    labels={'pickup_date_month': 'Month',
                                           'market_share': 'Market Share (%)',
                                           'vehicle.make': 'Brand'})
            fig_brand_share.update_layout(height=400)
            st.plotly_chart(fig_brand_share)
            
            # Add a summary table
            st.write(f"Summary for {city}:")
            summary = city_data.groupby('vehicle.make').agg({
                'rentals': 'sum',
                'market_share': 'mean'
            }).sort_values('rentals', ascending=False)
            summary.columns = ['Total Rentals', 'Average Market Share (%)']
            summary['Average Market Share (%)'] = summary['Average Market Share (%)'].round(2)
            st.dataframe(summary)
    else:
        st.warning("Please select at least one city to analyze.")

    # 2. Customer Loyalty Analysis
    st.subheader("2. Customer Rental Frequency and Value")
    
    # Calculate customer metrics
    customer_metrics = df_time.groupby('customer.id').agg({
        'rental_duration': ['count', 'mean'],
        'revenue': 'sum',
        'rating': 'mean'
    }).reset_index()
    customer_metrics.columns = ['customer_id', 'rental_count', 'avg_duration', 'total_revenue', 'avg_rating']
    
    # Create scatter plot
    fig_customer = px.scatter(customer_metrics,
                             x='rental_count',
                             y='total_revenue',
                             size='avg_duration',
                             color='avg_rating',
                             title='Customer Value Analysis',
                             labels={'rental_count': 'Number of Rentals',
                                    'total_revenue': 'Total Revenue ($)',
                                    'avg_duration': 'Average Rental Duration',
                                    'avg_rating': 'Average Rating'})
    st.plotly_chart(fig_customer)

    # Add customer segments
    st.write("Customer Segments Analysis:")
    customer_metrics['value_segment'] = pd.qcut(customer_metrics['total_revenue'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    segment_summary = customer_metrics.groupby('value_segment').agg({
        'customer_id': 'count',
        'rental_count': 'mean',
        'avg_duration': 'mean',
        'total_revenue': 'mean',
        'avg_rating': 'mean'
    }).round(2)
    segment_summary.columns = ['Number of Customers', 'Avg Rentals', 'Avg Duration', 'Avg Revenue', 'Avg Rating']
    st.dataframe(segment_summary)

    # 3. Vehicle Type Performance Analysis
    st.subheader("3. Vehicle Type Performance Metrics")
    
    vehicle_metrics = df_time.groupby('vehicle.type').agg({
        'rental_duration': ['count', 'mean'],
        'revenue': ['sum', 'mean'],
        'rating': 'mean'
    }).reset_index()
    vehicle_metrics.columns = ['vehicle_type', 'rental_count', 'avg_duration', 'total_revenue', 'avg_revenue', 'avg_rating']
    
    # Create radar chart
    categories = ['Rental Count', 'Avg Duration', 'Total Revenue', 'Avg Revenue', 'Avg Rating']
    fig_radar = go.Figure()
    
    for v_type in vehicle_metrics['vehicle_type'].unique():
        values = vehicle_metrics[vehicle_metrics['vehicle_type'] == v_type].iloc[0]
        # Normalize values
        normalized_values = [
            values['rental_count'] / vehicle_metrics['rental_count'].max(),
            values['avg_duration'] / vehicle_metrics['avg_duration'].max(),
            values['total_revenue'] / vehicle_metrics['total_revenue'].max(),
            values['avg_revenue'] / vehicle_metrics['avg_revenue'].max(),
            values['avg_rating'] / vehicle_metrics['avg_rating'].max(),
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],
            theta=categories + [categories[0]],
            name=v_type
        ))
    
    fig_radar.update_layout(title='Vehicle Type Performance Comparison')
    st.plotly_chart(fig_radar)

    # 4. Price Sensitivity Analysis
    st.subheader("4. Price Sensitivity by Vehicle Type")
    
    # Calculate booking rate at different price points
    price_sensitivity = df_time.groupby(['vehicle.type', pd.qcut(df_time['rate.daily'], q=5)]).agg({
        'rental_duration': 'count',
        'revenue': 'sum'
    }).reset_index()
    price_sensitivity.columns = ['vehicle_type', 'price_range', 'bookings', 'revenue']
    
    fig_price = px.line(price_sensitivity,
                        x='price_range',
                        y='bookings',
                        color='vehicle_type',
                        title='Booking Volume by Price Range',
                        labels={'price_range': 'Daily Rate Range',
                               'bookings': 'Number of Bookings',
                               'vehicle_type': 'Vehicle Type'})
    st.plotly_chart(fig_price)

    # 5. Rental Duration Patterns
    st.subheader("5. Rental Duration Patterns by Vehicle Type and Season")
    
    duration_patterns = df_time.groupby(['vehicle.type', 'pickup_date_month'])['rental_duration'].mean().reset_index()
    
    fig_duration = px.line(duration_patterns,
                          x='pickup_date_month',
                          y='rental_duration',
                          color='vehicle.type',
                          title='Average Rental Duration by Month and Vehicle Type',
                          labels={'pickup_date_month': 'Month',
                                 'rental_duration': 'Average Duration (Days)',
                                 'vehicle.type': 'Vehicle Type'})
    st.plotly_chart(fig_duration)

    # 6. City-to-City Flow Analysis
    st.subheader("6. Popular Rental Routes")
    
    route_analysis = df.groupby(['pickUp.city', 'dropOff.city']).size().reset_index(name='flow')
    route_analysis = route_analysis.sort_values('flow', ascending=False).head(20)
    
    fig_routes = px.bar(route_analysis,
                       x='pickUp.city',
                       y='flow',
                       color='dropOff.city',
                       title='Top 20 Popular Rental Routes',
                       labels={'pickUp.city': 'Pickup City',
                              'flow': 'Number of Rentals',
                              'dropOff.city': 'Drop-off City'})
    st.plotly_chart(fig_routes)

    # 7. Vehicle Age Impact Analysis
    st.subheader("7. Vehicle Age Impact on Performance")
    
    current_year = pd.to_datetime('now').year
    df_time['vehicle_age'] = current_year - df_time['vehicle.year']
    
    age_impact = df_time.groupby('vehicle_age').agg({
        'rate.daily': 'mean',
        'rating': 'mean',
        'rental_duration': 'count'
    }).reset_index()
    
    fig_age = go.Figure()
    fig_age.add_trace(go.Scatter(x=age_impact['vehicle_age'], y=age_impact['rate.daily'],
                                name='Average Daily Rate', yaxis='y1'))
    fig_age.add_trace(go.Scatter(x=age_impact['vehicle_age'], y=age_impact['rating'],
                                name='Average Rating', yaxis='y2'))
    fig_age.add_trace(go.Bar(x=age_impact['vehicle_age'], y=age_impact['rental_duration'],
                            name='Number of Rentals', yaxis='y3'))
    
    fig_age.update_layout(
        title='Impact of Vehicle Age on Performance Metrics',
        yaxis=dict(title='Average Daily Rate ($)', side='left'),
        yaxis2=dict(title='Average Rating', side='right', overlaying='y'),
        yaxis3=dict(title='Number of Rentals', side='right', overlaying='y', position=0.85)
    )
    st.plotly_chart(fig_age)

    # 8. Fuel Type Trends
    st.subheader("8. Fuel Type Popularity Trends")
    
    fuel_trends = df_time.groupby(['pickup_date_month', 'fuelType']).agg({
        'rental_duration': 'count',
        'revenue': 'sum',
        'rating': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_fuel_rentals = px.line(fuel_trends,
                                  x='pickup_date_month',
                                  y='rental_duration',
                                  color='fuelType',
                                  title='Rental Volume by Fuel Type',
                                  labels={'pickup_date_month': 'Month',
                                         'rental_duration': 'Number of Rentals',
                                         'fuelType': 'Fuel Type'})
        st.plotly_chart(fig_fuel_rentals)
    
    with col2:
        fig_fuel_revenue = px.line(fuel_trends,
                                  x='pickup_date_month',
                                  y='revenue',
                                  color='fuelType',
                                  title='Revenue by Fuel Type',
                                  labels={'pickup_date_month': 'Month',
                                         'revenue': 'Total Revenue ($)',
                                         'fuelType': 'Fuel Type'})
        st.plotly_chart(fig_fuel_revenue)

    # 9. Weekend vs Weekday Analysis
    st.subheader("9. Weekend vs Weekday Rental Patterns")
    
    day_patterns = df_time.groupby(['pickup_date_is_weekend', 'vehicle.type']).agg({
        'rental_duration': ['count', 'mean'],
        'rate.daily': 'mean',
        'revenue': 'mean'
    }).reset_index()
    day_patterns.columns = ['is_weekend', 'vehicle_type', 'rental_count', 'avg_duration', 'avg_rate', 'avg_revenue']
    day_patterns['period'] = day_patterns['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    
    fig_weekend = px.bar(day_patterns,
                        x='vehicle_type',
                        y=['rental_count', 'avg_duration', 'avg_rate', 'avg_revenue'],
                        facet_col='period',
                        title='Weekend vs Weekday Rental Patterns by Vehicle Type',
                        labels={'vehicle_type': 'Vehicle Type',
                               'value': 'Metric Value',
                               'variable': 'Metric'})
    st.plotly_chart(fig_weekend)

    # 10. Customer Review Analysis
    st.subheader("10. Customer Satisfaction Analysis")
    
    review_analysis = df_time.groupby(['vehicle.make', 'vehicle.type']).agg({
        'rating': ['mean', 'count', 'std'],
        'reviewCount': 'sum'
    }).reset_index()
    review_analysis.columns = ['make', 'type', 'avg_rating', 'rental_count', 'rating_std', 'review_count']
    
    fig_reviews = px.scatter(review_analysis,
                            x='avg_rating',
                            y='rental_count',
                            size='review_count',
                            color='type',
                            hover_data=['make', 'rating_std'],
                            title='Customer Satisfaction vs Popularity',
                            labels={'avg_rating': 'Average Rating',
                                   'rental_count': 'Number of Rentals',
                                   'review_count': 'Number of Reviews',
                                   'type': 'Vehicle Type'})
    st.plotly_chart(fig_reviews)

def main():
    st.title("Advanced Business Analytics Dashboard")

    df = load_data()
    
    if df is not None:
        # Display data info
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Date range: {pd.to_datetime(df['pickup_date']).min().date()} to {pd.to_datetime(df['pickup_date']).max().date()}")
        with col2:
            st.write("Analysis includes:")
            st.write("- Brand popularity trends")
            st.write("- Customer behavior patterns")
            st.write("- Vehicle performance metrics")
        
        # Perform advanced analytics
        analyze_advanced_metrics(df)
    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
