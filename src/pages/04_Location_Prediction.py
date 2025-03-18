import streamlit as st
import pandas as pd
import plotly.express as px
from models.location_predictor import LocationPredictor
import os

st.set_page_config(page_title="Location Demand Prediction", layout="wide")

def load_data():
    """Load and preprocess the car rental data"""
    data_path = os.path.join('data', 'processed_car_rental.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['pickup_date'])
        df['dropoff_date'] = pd.to_datetime(df['dropoff_date'])
        return df
    return None

def main():
    st.title("🎯 Location Demand Prediction")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Model Training Section
        st.header("1. Model Training")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Overview:")
            st.write(f"Number of records: {df.shape[0]:,}")
            st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        with col2:
            st.write("Features used:")
            st.write("- Temporal: Month, Day of Week, Weekend flag")
            st.write("- Vehicle: Type, Historical performance")
            st.write("- Location: Historical demand, ratings")
        
        if st.button("Train Model"):
            try:
                with st.spinner("Training models..."):
                    # Validate required columns
                    required_cols = ['pickUp.city', 'vehicle.type', 'pickup_date', 'rate.daily', 'rating']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        return
                    
                    # Initialize and train model
                    predictor = LocationPredictor()
                    metrics = predictor.train(df)
                    
                    st.success("Models trained successfully!")
                    
                    # Display detailed metrics
                    st.subheader("Model Performance")
                    
                    # Demand prediction metrics
                    st.write("Demand Prediction Model:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{metrics['demand_metrics']['r2_score']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['demand_metrics']['rmse']:.1f}")
                    with col3:
                        # Show top feature importance
                        top_feature = max(metrics['demand_metrics']['feature_importance'].items(), key=lambda x: x[1])
                        st.metric("Top Feature", top_feature[0])
                    
                    # Vehicle type metrics
                    st.write("\nVehicle Type Recommendation Model:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{metrics['vehicle_metrics']['accuracy']:.3f}")
                    with col2:
                        st.text("Classification Report:")
                        st.text(metrics['vehicle_metrics']['classification_report'])
                    
                    # Save model
                    model_dir = 'models'
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    predictor.save_model(os.path.join(model_dir, 'location_predictor.joblib'))
                    
                    st.success("Model saved successfully! You can now make predictions.")
                    
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.error("Please ensure the data format is correct and try again.")
        
        # Prediction Section
        st.header("2. Future Predictions")
        
        locations = sorted(df['pickUp.city'].unique())
        vehicle_types = sorted(df['vehicle.type'].unique())
        
        try:
            predictor = LocationPredictor()
            predictor.load_model('models/location_predictor.joblib')
            
            # Location and date selection
            selected_location = st.selectbox("Select location:", locations)
            selected_date = st.date_input(
                "Select prediction date:",
                value=pd.Timestamp.now() + pd.Timedelta(days=30)
            )
            
            # Check if model exists and is trained
            model_path = 'models/location_predictor.joblib'
            if not os.path.exists(model_path):
                st.warning("Please train the model first before making predictions.")
                return
            
            try:
                # Validate model is trained
                if not hasattr(predictor.vehicle_model, 'classes_') or not hasattr(predictor.vehicle_type_encoder, 'classes_'):
                    st.error("The model needs to be trained first. Please use the 'Train Model' button above.")
                    return
                
                # Get recommendations
                recommendations = predictor.recommend_vehicle_types(
                    selected_location,
                    pd.Timestamp(selected_date)
                )
                
                # Display recommendations
                st.subheader("Vehicle Type Recommendations")
                
                # Show top recommendations with confidence scores
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_rec = px.bar(
                        recommendations,
                        x='vehicle_type',
                        y='confidence',
                        title=f'Recommended Vehicle Types for {selected_location}',
                        labels={'confidence': 'Confidence Score', 'vehicle_type': 'Vehicle Type'}
                    )
                    fig_rec.update_layout(
                        xaxis_tickangle=45,
                        yaxis_tickformat='.1%',
                        height=400
                    )
                    st.plotly_chart(fig_rec)
                
                with col2:
                    st.subheader("Top Recommendations")
                    for _, row in recommendations.head(3).iterrows():
                        confidence = row['confidence'] * 100
                        emoji = "🟢" if confidence > 80 else "🟡" if confidence > 50 else "🔴"
                        st.write(f"{emoji} {row['vehicle_type']}: {confidence:.1f}% confidence")
                
            except Exception as e:
                st.error(f"Error getting vehicle recommendations: {str(e)}")
                st.error("Please ensure the model is trained with the correct data format.")
                return
            
            try:
                # Validate model is trained
                if not hasattr(predictor.demand_model, 'feature_importances_'):
                    st.error("The demand prediction model needs to be trained first. Please use the 'Train Model' button above.")
                    return

                # Generate future dates
                future_dates = [
                    pd.Timestamp(selected_date) + pd.DateOffset(months=i)
                    for i in range(12)
                ]
                
                # Get demand predictions
                predictions = predictor.predict_future_demand(
                    future_dates,
                    [selected_location],
                    vehicle_types,
                    df
                )
                
                # Display demand forecast
                st.subheader("Demand Forecast")
                
                # Create main demand forecast plot
                fig_demand = px.line(
                    predictions,
                    x='date',
                    y='predicted_demand',
                    color='vehicle_type',
                    title=f'12-Month Demand Forecast for {selected_location}',
                    labels={
                        'predicted_demand': 'Predicted Demand',
                        'date': 'Month',
                        'vehicle_type': 'Vehicle Type'
                    }
                )
                fig_demand.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Predicted Demand",
                    height=500,
                    legend_title="Vehicle Type",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_demand)
                
                # Display insights
                st.header("3. Key Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Peak Demand Periods")
                    monthly_avg = predictions.groupby(predictions['date'].dt.strftime('%B'))['predicted_demand'].mean()
                    peak_months = monthly_avg.nlargest(3)
                    
                    for month, demand in peak_months.items():
                        st.write(f"📅 {month}: {demand:.1f} average demand")
                    
                    # Add trend analysis
                    trend = "increasing" if monthly_avg.iloc[-1] > monthly_avg.iloc[0] else "decreasing"
                    st.write(f"\n📈 Overall trend: Demand is {trend} over the forecast period")
                
                with col2:
                    st.subheader("Vehicle Type Analysis")
                    vehicle_avg = predictions.groupby('vehicle_type')['predicted_demand'].mean().sort_values(ascending=False)
                    
                    for vtype, demand in vehicle_avg.head(3).items():
                        st.write(f"🚗 {vtype}: {demand:.1f} average demand")
                    
                    # Add seasonal insight
                    peak_season = peak_months.index[0]
                    st.write(f"\n🌟 Peak season: {peak_season}")
            
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.error("Please ensure the model is trained with the correct data format.")
                return
            

            
        except FileNotFoundError:
            st.warning("Please train the model first.")
    
    else:
        st.error("Could not find data file: 'data/processed_car_rental.csv'")

if __name__ == "__main__":
    main()
