import os
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import timedelta
from models.location_predictor import LocationPredictor

from analytics.HyperParameter_Tuning.XGBTuner import XGBTuner

# Configuration
st.set_page_config(page_title="Location Demand Prediction", layout="wide")

DATA_PATH = os.path.join("data", "processed_car_rental.csv")
MODEL_PATH = os.path.join("models", "vehicle_recommender.joblib")


# Utility Functions
def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the car rental data from a CSV file.
    """
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert relevant columns to datetime
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df["dropoff_date"] = pd.to_datetime(df["dropoff_date"], errors="coerce")  # In case some rows are invalid

    # Rename columns for consistency
    df.rename(columns={
        "pickUp.city": "pickup_city",
        "vehicle.type": "vehicle_type"
    }, inplace=True)

    # Add a standard 'date' column to unify reference
    if "date" not in df.columns:
        df["date"] = df["pickup_date"]

    return df


def display_dataset_info(df: pd.DataFrame):
    """
    Displays a quick overview of the dataset: number of records, date range, and used columns.
    """
    st.write("Dataset Overview:")
    st.write(f"Number of records: {df.shape[0]:,}")
    date_min = df['date'].min().date()
    date_max = df['date'].max().date()
    st.write(f"Date range: {date_min} to {date_max}")


def display_feature_info():
    """
    Displays the primary features used in the model.
    """
    st.write("Features used:")
    st.write("- Temporal: Month, Day of Week, Weekend flag")
    st.write("- Vehicle: Type, Historical performance")
    st.write("- Location: Historical demand, ratings")


def train_model(df: pd.DataFrame, predictor: LocationPredictor) -> None:
    """
    Trains the vehicle type recommendation model,
    displays training metrics,
    and saves the trained models to disk.
    """
    required_cols = ["pickup_city", "vehicle_type", "pickup_date"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    try:
        # ------------------------------
        # 1. Hyperparameter Tuning Step
        # ------------------------------

        st.info("Starting hyperparameter tuning...")
        tuner = XGBTuner()

        predictor.fit_encoders(df)
        with st.spinner("Tuning hyperparameters..."):
            tuner.tune_model(predictor, df, cv=5)
        st.success("Hyperparameter tuning complete!")

        # ------------------------------
        # 2. Train the Model with Tuned Parameters
        # ------------------------------
        with st.spinner("Training models..."):
            predictor.train(df)
        st.success("Models trained successfully!")

        # ------------------------------
        # 3. Save the Final Model
        # ------------------------------
        predictor.save_model(MODEL_PATH)
        st.success("Model saved successfully! You can now make predictions.")

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.error("Please ensure the data format is correct and try again.")


def display_model_metrics(df: pd.DataFrame, predictor: LocationPredictor) -> None:
    """
    Displays the metrics of the trained model.
    """

    st.subheader("Model Performance")
    # Show the trained model parameters
    st.write(predictor.vehicle_model.get_params())

    vehicle_metrics = predictor.evaluate(df)

    st.write("**Vehicle Type Recommendation Model**:")
    c1, c2 = st.columns(2)
    with c1:
        # Correctly access accuracy from the nested dictionary
        st.metric("Accuracy", f"{vehicle_metrics['vehicle_metrics']['accuracy']:.3f}")
    with c2:
        st.text("Classification Report:")
        st.text(vehicle_metrics['vehicle_metrics']['classification_report'])


def predict_future(df: pd.DataFrame, predictor: LocationPredictor) -> None:
    """
    Loads the trained model, then provides:
     - Vehicle type recommendations for a user-selected date and location.
     - (Optionally) any other forecast or analysis you'd like to add.
    """
    if not os.path.exists(MODEL_PATH + "_xgb_model.json"):
        st.warning("Please train the model before making predictions.")
        return

    predictor.load_model(MODEL_PATH)

    # Convert columns for consistency
    if "pickup_city" in df.columns:
        df.rename(columns={"pickup_city": "location"}, inplace=True)
    if "vehicle_type" in df.columns:
        df.rename(columns={"vehicle_type": "vehicle_type"}, inplace=True)

    # Prepare location and vehicle type lists
    locations = sorted(df["location"].dropna().unique().tolist())

    st.subheader("Vehicle Recommendations")
    selected_location = st.selectbox("Select Location:", locations)
    selected_date = st.date_input(
        "Select Prediction Date:",
        pd.Timestamp.now() + pd.Timedelta(days=30)
    )

    # If the vehicle model has not been trained or loaded, stop
    if not hasattr(predictor.vehicle_model, "classes_"):
        st.error("The vehicle type model is not trained. Please train it first.")
        return

    try:
        # Make recommendations
        recommendations = predictor.recommend_vehicle_types(
            selected_location,
            pd.Timestamp(selected_date)
        )

        st.subheader("Recommended Vehicle Types")
        colA, colB = st.columns([2, 1])
        with colA:
            fig_rec = px.bar(
                recommendations,
                x='vehicle_type',
                y='confidence',
                title=f"Recommendations for {selected_location}",
                labels={"confidence": "Confidence Score", "vehicle_type": "Vehicle Type"}
            )
            fig_rec.update_layout(
                xaxis_tickangle=45,
                yaxis_tickformat='.1%',
                height=400
            )
            st.plotly_chart(fig_rec)

        with colB:
            st.subheader("Top 3")
            for _, row in recommendations.head(3).iterrows():
                confidence = row["confidence"] * 100
                if confidence > 80:
                    emoji = "🟢"
                elif confidence > 50:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                st.write(f"{emoji} {row['vehicle_type']}: {confidence:.1f}%")

    except Exception as e:
        st.error(f"Error getting vehicle recommendations: {str(e)}")

    # Demand forecast
    st.subheader("Demand Forecast")

    # Create / rename columns if needed
    df["pickup_date_year"] = df["pickup_date"].dt.year

    # Group to create quarterly demand
    demand_df = (
        df.groupby(["location", "vehicle_type", "pickup_date_year", "pickup_date_quarter"])
        .size()
        .reset_index(name="demand")
    )

    df = df.merge(demand_df, on=["location", "vehicle_type", "pickup_date_year", "pickup_date_quarter"], how="left")

    vehicle_types = sorted(df["vehicle_type"].dropna().unique().tolist())

    # Get seasonal demand predictions
    try:
        predictions = predictor.predict_seasonal_demand(
            df,
            selected_location,
            vehicle_types
        )
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return

    fig_demand = px.line(
        predictions,
        x="season",
        y="predicted_demand",
        color="vehicle_type",
        title=f"Seasonal Demand Forecast for {selected_location}",
        labels={
            "season": "Season",
            "predicted_demand": "Predicted Demand",
            "vehicle_type": "Vehicle Type"
        }
    )
    fig_demand.update_layout(
        xaxis_title="Season",
        yaxis_title="Predicted Demand",
        height=500,
        legend_title="Vehicle Type",
        hovermode="x unified"
    )
    st.plotly_chart(fig_demand)

    # Key insights
    st.header("Key Insights")
    colC, colD = st.columns(2)

    with colC:
        st.subheader("Peak Demand Periods")
        seasonal_avg = predictions.groupby("season")["predicted_demand"].mean()
        peak_seasons = seasonal_avg.nlargest(3)

        for sname, val in peak_seasons.items():
            st.write(f"📅 {sname}: {val:.1f} average demand")

        # Simple trend analysis
        possible_seasons = ["Spring", "Summer", "Autumn", "Winter"]
        seasonal_avg_sorted = seasonal_avg.reindex(possible_seasons).dropna()
        if len(seasonal_avg_sorted) >= 2:
            first_val = seasonal_avg_sorted.iloc[0]
            last_val = seasonal_avg_sorted.iloc[-1]
            trend = "increasing" if last_val > first_val else "decreasing"
            st.write(
                f"📈 Overall trend: Demand is {trend} from {seasonal_avg_sorted.index[0]} to {seasonal_avg_sorted.index[-1]}")
        else:
            st.write("📈 Not enough seasonal data to determine a clear trend.")

    with colD:
        st.subheader("Vehicle Type Analysis")
        vehicle_avg = (
            predictions.groupby("vehicle_type")["predicted_demand"]
            .mean()
            .sort_values(ascending=False)
        )
        for vtype, val in vehicle_avg.head(3).items():
            st.write(f"🚗 {vtype}: {val:.1f} average demand")

        if not peak_seasons.empty:
            st.write(f"\n🌟 Peak season: {peak_seasons.index[0]}")
        else:
            st.write("🌟 No peak season identified (not enough data).")


# Main
def main():
    st.title("🎯 Location Demand Prediction")

    df = load_data(DATA_PATH)
    if df is None:
        st.error(f"Could not find data file at '{DATA_PATH}'. Please place a valid CSV there.")
        return

    # Section 1: Model Training
    st.header("1. Model Training")
    col1, col2 = st.columns(2)
    with col1:
        display_dataset_info(df)
    with col2:
        display_feature_info()

    # Load predictor instance
    predictor = LocationPredictor()

    # Check if model already exists and load it
    if os.path.exists(MODEL_PATH + "_xgb_model.json"):
        st.warning("A trained model already exists. If you want to retrain, please reset or delete the existing model.")
        predictor.load_model(MODEL_PATH)
        display_model_metrics(df, predictor)  # Show existing model metrics
    else:
        if st.button("Train Model"):
            train_model(df, predictor)
            display_model_metrics(df, predictor)

    # Section 2: Future Predictions
    st.header("2. Future Predictions")
    predict_future(df, predictor)


if __name__ == "__main__":
    main()
