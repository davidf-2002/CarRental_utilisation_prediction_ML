import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

class LocationPredictor:
    """
    A class for training and using two models:
      1) A RandomForestRegressor to predict demand.
      2) A RandomForestClassifier to recommend vehicle types.
    """

    def __init__(self):
        # Models
        self.vehicle_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)

        # Encoders and Scaler
        self.location_encoder = LabelEncoder()
        self.vehicle_type_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Internal feature importance cache
        self.feature_importance = {}

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Trains the vehicle type recommendation model,
        returning a dictionary of performance metrics.
        """

        # Validate required columns
        required_cols = ["pickup_city", "vehicle_type", "pickup_date", "rate.daily", "rating"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {', '.join(missing_cols)}")

        # Convert pickup_date to datetime
        df["pickup_date"] = pd.to_datetime(df["pickup_date"])

        # Prepare features for vehicle classification
        X_vehicle, y_vehicle = self._prepare_vehicle_data(df)
        X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vehicle, y_vehicle, test_size=0.2, random_state=42)
        self.vehicle_model.fit(X_train_v, y_train_v)

        # Evaluate vehicle classification model
        y_pred_v = self.vehicle_model.predict(X_test_v)
        vehicle_metrics = {
            "accuracy": accuracy_score(y_test_v, y_pred_v),
            "classification_report": classification_report(
                y_test_v, y_pred_v, target_names=self.vehicle_type_encoder.classes_
            )
        }

        return {
            "vehicle_metrics": vehicle_metrics
        }

    def _prepare_vehicle_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Internal helper to extract features and target for vehicle type classification.
        """

        df["pickup_city_encoded"] = self.location_encoder.fit_transform(df["pickup_city"])
        df["month"] = df["pickup_date"].dt.month
        df["day_of_week"] = df["pickup_date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["vehicle_type_encoded"] = self.vehicle_type_encoder.fit_transform(df["vehicle_type"])

        # We treat vehicle_type_encoded as a target, so it won't go into the features
        feature_cols = ["pickup_city_encoded", "month", "day_of_week", "is_weekend", "rate.daily", "rating"]
        X = df[feature_cols].fillna(0)
        y = df["vehicle_type_encoded"]

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=feature_cols)
        return X_scaled, y

    def predict_seasonal_demand(
        self,
        df: pd.DataFrame,
        location: str,
        vehicle_types: List[str]
    ) -> pd.DataFrame:
        """
        Predicts seasonal demand for a particular location across multiple vehicle types.
        """

        # Hard-coded next seasons
        future_seasons = ["Spring", "Summer", "Autumn", "Winter"]

        # Ensure date column is standardised
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        elif "pickup_date" in df.columns:
            df["date"] = pd.to_datetime(df["pickup_date"], dayfirst=True, errors="coerce")
        else:
            raise ValueError("DataFrame must have a 'date' or 'pickup_date' column.")

        # Filter for selected location
        df_loc = df[df["location"] == location].copy()
        df_loc.dropna(subset=["demand", "vehicle_type", "pickup_date_quarter"], inplace=True)

        quarter_to_season = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
        df_loc["season"] = df_loc["pickup_date_quarter"].map(quarter_to_season)

        # Group historical demand
        grouped = (
            df_loc.groupby(["vehicle_type", "season"], as_index=False)
            .agg({"demand": "mean"})
            .rename(columns={"demand": "avg_historical_demand"})
        )

        # Prepare training set
        season_map = {season: i for i, season in enumerate(["Spring", "Summer", "Autumn", "Winter"])}
        grouped["season_num"] = grouped["season"].map(season_map)

        unique_vtypes = grouped["vehicle_type"].unique().tolist()
        vtype_map = {v: i for i, v in enumerate(unique_vtypes)}
        grouped["vehicle_type_num"] = grouped["vehicle_type"].map(vtype_map)

        X_train = grouped[["season_num", "vehicle_type_num"]]
        y_train = grouped["avg_historical_demand"]

        if len(X_train) < 2:
            logging.warning("Not enough data to train a reliable seasonal model.")
            return pd.DataFrame(columns=["season", "location", "vehicle_type", "predicted_demand"])

        # Train a lightweight model for demonstration
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict for upcoming seasons
        future_combos = [(s, vt) for s in future_seasons for vt in vehicle_types]
        future_df = pd.DataFrame(future_combos, columns=["season", "vehicle_type"])
        future_df["season_num"] = future_df["season"].map(season_map)
        future_df["vehicle_type_num"] = future_df["vehicle_type"].map(vtype_map).fillna(-1)

        X_pred = future_df[["season_num", "vehicle_type_num"]]
        preds = model.predict(X_pred)

        future_df["predicted_demand"] = preds
        future_df["location"] = location
        future_df.sort_values(["season", "vehicle_type"], inplace=True)
        future_df = future_df[["season", "location", "vehicle_type", "predicted_demand"]]

        return future_df


    def recommend_vehicle_types(self, location: str, date: pd.Timestamp) -> pd.DataFrame:
        """
        Recommends vehicle types for a given location and date using the trained classifier.
        """
        if not location or not date:
            raise ValueError("Location and date must be provided.")
        if not hasattr(self.vehicle_model, "classes_"):
            raise ValueError("Vehicle recommendation model is not trained.")

        # Prepare a small DataFrame with a row per possible vehicle type
        input_data = [{
            "pickup_date": date,
            "pickup_city": location,
            "rate.daily": 100.0,   # Example default rate
            "rating": 4.0,         # Example default rating
            "vehicle_type": vt
        } for vt in self.vehicle_type_encoder.classes_]

        df_input = pd.DataFrame(input_data)
        df_input["pickup_city_encoded"] = self.location_encoder.transform([location] * len(df_input))
        df_input["vehicle_type_encoded"] = self.vehicle_type_encoder.transform(df_input["vehicle_type"])

        # Extract features
        X_input, _ = self._prepare_vehicle_data(df_input)

        # Get probabilities
        probs = self.vehicle_model.predict_proba(X_input)

        # Build recommendations
        recs = pd.DataFrame({
            "vehicle_type": self.vehicle_type_encoder.classes_,
            "confidence": [p[1] for p in probs]
        })
        recs = recs.sort_values("confidence", ascending=False)
        return recs

    def save_model(self, path: str) -> None:
        """
        Saves model components to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "vehicle_model": self.vehicle_model,
            "location_encoder": self.location_encoder,
            "vehicle_type_encoder": self.vehicle_type_encoder,
            "scaler": self.scaler,
            "feature_importance": self.feature_importance,
        }, path)

    def load_model(self, path: str) -> None:
        """
        Loads model components from the specified file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        components = joblib.load(path)
        self.vehicle_model = components["vehicle_model"]
        self.location_encoder = components["location_encoder"]
        self.vehicle_type_encoder = components["vehicle_type_encoder"]
        self.scaler = components["scaler"]
        self.feature_importance = components["feature_importance"]
