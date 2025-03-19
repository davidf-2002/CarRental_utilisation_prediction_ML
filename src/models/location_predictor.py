import os
import joblib
import numpy as np
import pandas as pd
import logging
import xgboost as xgb
from typing import Dict, List, Tuple, Any
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
    def __init__(self):
        self.vehicle_model = None
        self.location_encoder = LabelEncoder()
        self.vehicle_type_encoder = LabelEncoder()
        # Keep track of the feature columns used to train
        self.feature_cols = [
            "pickup_city_encoded",
            "pickup_date_dayofweek",
            "pickup_date_month",
            "pickup_date_is_weekend"
        ]

    def _prepare_vehicle_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        Prepare the features (X) and target (y) for vehicle type classification,
        including encoding of categorical columns and feature engineering.
        """

        # 1) Encode location and vehicle type
        #    (Location encoding is for use as a feature; vehicle type is the target)
        if "pickup_city_encoded" not in df.columns:
            df["pickup_city_encoded"] = self.location_encoder.transform(df["pickup_city"])

        if "vehicle_type_encoded" not in df.columns:
            df["vehicle_type_encoded"] = self.vehicle_type_encoder.transform(df["vehicle_type"])

        X = df[self.feature_cols].copy()
        y = df["vehicle_type_encoded"].copy()
        return X, y

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a multi-class vehicle type recommendation model using an XGBoost classifier.
        Returns a dictionary containing model performance metrics.
        """
        # Validate required columns
        required_cols = ["pickup_city", "pickup_date"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {', '.join(missing_cols)}")

        # Fit label encoders on the entire dataset for consistent encoding
        self.location_encoder.fit(df["pickup_city"])
        self.vehicle_type_encoder.fit(df["vehicle_type"])

        # Add date-related features before model preparation
        df["pickup_date_dayofweek"] = df["pickup_date"].dt.dayofweek
        df["pickup_date_month"] = df["pickup_date"].dt.month
        df["pickup_date_is_weekend"] = df["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        # Now that the columns are added, check if they're in the DataFrame
        missing_date_cols = ["pickup_date_dayofweek", "pickup_date_month", "pickup_date_is_weekend"]
        for col in missing_date_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col} in DataFrame.")

        # Prepare data for training
        X, y = self._prepare_vehicle_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create & train XGBoost classifier
        self.vehicle_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,  # Avoids XGBoost's own label-encoding warning
            eval_metric="mlogloss"
        )
        self.vehicle_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.vehicle_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        clf_report = classification_report(
            y_test,
            y_pred,
            target_names=self.vehicle_type_encoder.classes_
        )

        vehicle_metrics = {
            "accuracy": acc,
            "classification_report": clf_report
        }

        return {
            "vehicle_metrics": vehicle_metrics
        }

    def recommend_vehicle_types(self, location: str, date: pd.Timestamp) -> pd.DataFrame:
        """
        Recommend vehicle types for a given location and date using the trained classifier.
        Produces a DataFrame of all vehicle types and their predicted confidence scores.
        """
        if not location or not date:
            raise ValueError("Location and date must be provided.")
        if not hasattr(self.vehicle_model, "classes_"):
            raise ValueError("Vehicle recommendation model is not trained.")

        # Build input data for each possible vehicle type
        vehicle_types_list = self.vehicle_type_encoder.classes_
        input_data = []
        for vt in vehicle_types_list:
            input_data.append({
                "pickup_date": date,
                "pickup_city": location,
                "vehicle_type": vt
            })

        df_input = pd.DataFrame(input_data)

        df_input["pickup_date_dayofweek"] = df_input["pickup_date"].dt.dayofweek
        df_input["pickup_date_month"] = df_input["pickup_date"].dt.month
        df_input["pickup_date_is_weekend"] = df_input["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        # Prepare features
        X_input, _ = self._prepare_vehicle_data(df_input)

        # Get predicted probabilities for each class
        probs = self.vehicle_model.predict_proba(X_input)  # shape => (N_rows, N_classes)

        # Because each row in df_input corresponds to a specific vehicle_type,
        # we extract the probability that the row’s vehicle type is indeed the correct class.
        #   i -> index in df_input
        #   c -> the encoded class for the vehicle type in that row
        confidences = []
        for i, row in df_input.iterrows():
            c = row["vehicle_type_encoded"]
            confidences.append(probs[i, c])

        # Build recommendations DataFrame
        recs = pd.DataFrame({
            "vehicle_type": df_input["vehicle_type"],
            "confidence": confidences
        }).sort_values("confidence", ascending=False)

        return recs

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

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model and encoders to disk.
        """
        # XGBoost-specific save
        self.vehicle_model.save_model(model_path + "_xgb_model.json")

        # Save encoders using pandas or joblib
        import joblib
        joblib.dump({
            "location_encoder": self.location_encoder,
            "vehicle_type_encoder": self.vehicle_type_encoder,
            "feature_cols": self.feature_cols
        }, model_path + "_encoders.pkl")

    def load_model(self, model_path: str) -> None:
        """
        Load the trained model and encoders from disk.
        """
        import joblib
        # Load XGBoost model
        self.vehicle_model = xgb.XGBClassifier()
        self.vehicle_model.load_model(model_path + "_xgb_model.json")

        # Load encoders
        data = joblib.load(model_path + "_encoders.pkl")
        self.location_encoder = data["location_encoder"]
        self.vehicle_type_encoder = data["vehicle_type_encoder"]
        self.feature_cols = data["feature_cols"]
