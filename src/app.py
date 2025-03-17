import streamlit as st
import pandas as pd
import plotly.express as px
import os
from preprocessing.preprocessing_utils import DataPreprocessor, extract_datetime_features
from models.model_trainer import ModelTrainer

st.set_page_config(
    page_title="Car Rental Analytics",
    page_icon="🚗",
    layout="wide"
)

def load_data():
    """Load and preprocess the car rental data"""
    try:
        data_path = os.path.join('data', 'processed_car_rental.csv')
        if not os.path.exists(data_path):
            st.error(f"Data file not found at {data_path}")
            return None
            
        # Read and validate data
        df = pd.read_csv(data_path)
        
        # Check required columns
        required_cols = ['pickup_date', 'dropoff_date', 'pickUp.city', 'vehicle.type', 'rate.daily', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Convert dates and ensure consistent column names
        df['date'] = pd.to_datetime(df['pickup_date'])
        df['dropoff_date'] = pd.to_datetime(df['dropoff_date'])
        
        # Handle missing values
        df['rating'] = df['rating'].fillna(df['rating'].mean())
        df['rate.daily'] = df['rate.daily'].fillna(df['rate.daily'].mean())
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("🚗 Car Rental Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display dataset info
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rentals", f"{df.shape[0]:,}")
            
        with col2:
            avg_rating = df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}⭐")
            
        with col3:
            total_revenue = (df['rate.daily'] * (pd.to_datetime(df['dropoff_date']) - pd.to_datetime(df['pickup_date'])).dt.days).sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        # Data Quality Check
        st.subheader("Data Quality")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning("Missing values detected in the dataset:")
            for col, count in missing_data[missing_data > 0].items():
                st.write(f"- {col}: {count:,} missing values")
        else:
            st.success("No missing values found in the dataset!")
        
        # Quick Stats
        st.subheader("Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Location stats
            location_stats = df['pickUp.city'].value_counts().head()
            fig_locations = px.bar(
                x=location_stats.index,
                y=location_stats.values,
                title="Top Rental Locations",
                labels={'x': 'Location', 'y': 'Number of Rentals'}
            )
            st.plotly_chart(fig_locations)
        
        with col2:
            # Vehicle type stats
            vehicle_stats = df['vehicle.type'].value_counts()
            fig_vehicles = px.pie(
                values=vehicle_stats.values,
                names=vehicle_stats.index,
                title="Vehicle Type Distribution"
            )
            st.plotly_chart(fig_vehicles)
        
        # Advanced Analysis Section
        st.header("General Analytics")
        
        # 1. Peak Rental Hours Analysis
        st.subheader("1. Peak Rental Hours Analysis")
        df_time = extract_datetime_features(df.copy(), 'pickup_date')
        
        col1, col2 = st.columns(2)
        with col1:
            # Hourly distribution
            hourly_rentals = df_time.groupby('pickup_date_hour').size().reset_index(name='count')
            fig_hourly = px.line(hourly_rentals, x='pickup_date_hour', y='count',
                               title='Rental Distribution by Hour of Day',
                               labels={'pickup_date_hour': 'Hour of Day', 'count': 'Number of Rentals'})
            st.plotly_chart(fig_hourly)
        
        with col2:
            # Day of week distribution
            daily_rentals = df_time.groupby('pickup_date_dayofweek').size().reset_index(name='count')

            # daily_rentals['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_map = {
                0: 'Monday',
                1: 'Tuesday',
                2: 'Wednesday',
                3: 'Thursday',
                4: 'Friday',
                5: 'Saturday',
                6: 'Sunday'
            }

            daily_rentals['day_name'] = daily_rentals['pickup_date_dayofweek'].map(day_map)

            fig_daily = px.bar(daily_rentals, x='day_name', y='count',
                             title='Rental Distribution by Day of Week',
                             labels={'day_name': 'Day of Week', 'count': 'Number of Rentals'})
            st.plotly_chart(fig_daily)

        # 2. Car Type Popularity by City
        st.subheader("2. Car Type Popularity by City")
        car_city_dist = df.groupby(['pickUp.city', 'vehicle.type']).size().reset_index(name='count')
        fig_heatmap = px.density_heatmap(car_city_dist, x='pickUp.city', y='vehicle.type', z='count',
                                       title='Car Type Popularity by City',
                                       labels={'pickUp.city': 'City', 'vehicle.type': 'Vehicle Type', 'count': 'Number of Rentals'})
        fig_heatmap.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_heatmap)

        # 3. Seasonal Trends
        st.subheader("3. Seasonal Trends")
        df_seasonal = extract_datetime_features(df.copy(), 'pickup_date')
        monthly_type_dist = df_seasonal.groupby(['pickup_date_month', 'vehicle.type']).size().reset_index(name='count')
        fig_seasonal = px.line(monthly_type_dist, x='pickup_date_month', y='count', color='vehicle.type',
                             title='Monthly Rental Trends by Vehicle Type',
                             labels={'pickup_date_month': 'Month', 'count': 'Number of Rentals', 'vehicle.type': 'Vehicle Type'})
        st.plotly_chart(fig_seasonal)

        # Model Training Section
        st.header("Model Training")
        # Select target column
        target_column = st.selectbox("Select target column", df.columns)
        
        if st.button("Train Model"):
            # Prepare data
            preprocessor = DataPreprocessor()
            df_cleaned = preprocessor.clean_data(df)
            X_scaled, y = preprocessor.prepare_features(df_cleaned, target_column)
            X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
            
            # Train model
            trainer = ModelTrainer()
            trainer.train(X_train, y_train)
            
            # Evaluate model
            evaluation = trainer.evaluate(X_test, y_test)
            
            # Display results
            st.subheader("Model Performance")
            st.write(f"Accuracy: {evaluation['accuracy']:.2f}")
            st.text("Classification Report:")
            st.text(evaluation['classification_report'])
            
            # Save model
            trainer.save_model()
            st.success("Model trained and saved successfully!")
            
            # Feature importance plot
            feature_importance = pd.DataFrame({
                'feature': X_scaled.columns,
                'importance': trainer.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance, x='feature', y='importance',
                        title='Feature Importance')
            st.plotly_chart(fig)
        
        # Save processed data
        df.to_csv('data/processed_car_rental.csv', index=False)
        
    else:
        st.error("Error: Could not find the data file at 'data/processed_car_rental.csv'")
        st.info("Please make sure the data file exists in the data directory.")

if __name__ == "__main__":
    main()
