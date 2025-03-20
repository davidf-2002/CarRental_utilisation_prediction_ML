import joblib
import pandas as pd
import logging
import xgboost as xgb
from typing import Dict, List, Any

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Configure logging for reproducibility and debugging
logging.basicConfig(level=logging.INFO)


class LocationPredictor:
    """
    LocationPredictor encapsulates vehicle type recommendation and seasonal demand forecasting.
    """
    def __init__(self):
        self.vehicle_model = None
        self.location_encoder = LabelEncoder()
        self.vehicle_type_encoder = LabelEncoder()
        # Keep track of the feature columns used to train the classifier
        self.feature_cols = [
            "pickup_city_encoded",
            "pickup_date_dayofweek",
            "pickup_date_month",
            "pickup_date_is_weekend"
        ]

    def _prepare_vehicle_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        Prepare the features (X) and target (y) for vehicle type classification.
        Performs encoding of categorical columns and feature engineering.
        """
        # Encode the pickup city for use as a feature
        if "pickup_city_encoded" not in df.columns:
            df["pickup_city_encoded"] = self.location_encoder.transform(df["pickup_city"])
        # Encode vehicle type as target variable
        if "vehicle_type_encoded" not in df.columns:
            df["vehicle_type_encoded"] = self.vehicle_type_encoder.transform(df["vehicle_type"])
        X = df[self.feature_cols].copy()
        y = df["vehicle_type_encoded"].copy()
        return X, y


    def train(self, df: pd.DataFrame) -> None:
        """
        Train an XGBoost classifier for vehicle type recommendation.
        This function only handles the training process.
        """
        # Validate required columns
        required_cols = ["pickup_city", "pickup_date", "vehicle_type"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {', '.join(missing_cols)}")

        # Fit label encoders on the full dataset
        self.location_encoder.fit(df["pickup_city"])
        self.vehicle_type_encoder.fit(df["vehicle_type"])

        # Create date-derived features
        df["pickup_date_dayofweek"] = df["pickup_date"].dt.dayofweek
        df["pickup_date_month"] = df["pickup_date"].dt.month
        df["pickup_date_is_weekend"] = df["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        # Prepare data for model training
        X, y = self._prepare_vehicle_data(df)

        # Split data into training and test sets (stratified for balanced classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Only initialise the model if it hasn't been set already (i.e. from tuning)
        if self.vehicle_model is None:
            self.vehicle_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )

        # Train the model on the training set
        self.vehicle_model.fit(self.X_train, self.y_train)

        # Save the trained model
        logging.info("Model training completed successfully.")

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the trained model's performance on the test set.
        This function performs predictions on the test set and returns the evaluation metrics.
        """
        if self.vehicle_model is None:
            raise ValueError("Model has not been trained. Please train the model first.")

        # Re-split the data for evaluation (using the full dataset)
        required_cols = ["pickup_city", "pickup_date", "vehicle_type"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for evaluation: {', '.join(missing_cols)}")

        # Fit label encoders
        self.location_encoder.fit(df["pickup_city"])
        self.vehicle_type_encoder.fit(df["vehicle_type"])

        # Create date-derived features
        df["pickup_date_dayofweek"] = df["pickup_date"].dt.dayofweek
        df["pickup_date_month"] = df["pickup_date"].dt.month
        df["pickup_date_is_weekend"] = df["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        # Prepare data for evaluation
        X, y = self._prepare_vehicle_data(df)

        # Re-split data into training and test sets (re-split for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Predict on the test set and evaluate
        y_pred = self.vehicle_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        clf_report = classification_report(
            y_test, y_pred, target_names=self.vehicle_type_encoder.classes_
        )
        cm = confusion_matrix(y_test, y_pred)

        logging.info("Model evaluation completed.")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info("Classification Report:\n" + clf_report)
        logging.info("Confusion Matrix:\n" + str(cm))

        return {
            "vehicle_metrics": {
                "accuracy": acc,
                "classification_report": clf_report,
                "confusion_matrix": cm
            }
        }

    def fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit the LabelEncoders on the entire dataset for consistent encoding.
        Call this before training or hyperparameter tuning.
        """
        df.rename(columns={
            "pickUp.city": "pickup_city",
            "vehicle.type": "vehicle_type"
        }, inplace=True)

        if "pickup_city" not in df.columns:
            raise ValueError("DataFrame must have 'pickup_city' column.")
        if "vehicle_type" not in df.columns:
            raise ValueError("DataFrame must have 'vehicle_type' column.")

        self.location_encoder.fit(df["pickup_city"])
        self.vehicle_type_encoder.fit(df["vehicle_type"])

    # def plot_learning_curve(self, df: pd.DataFrame, title: str = "Learning Curve") -> None:
    #     """
    #     Plot the learning curve for the XGBoost classifier.
    #     Helps in visualising model performance (training vs cross-validation) as data increases.
    #     """
    #     # Ensure date features are present
    #     df["pickup_date_dayofweek"] = df["pickup_date"].dt.dayofweek
    #     df["pickup_date_month"] = df["pickup_date"].dt.month
    #     df["pickup_date_is_weekend"] = df["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    #     X, y = self._prepare_vehicle_data(df)
    #
    #     train_sizes, train_scores, test_scores = learning_curve(
    #         self.vehicle_model, X, y, cv=5, scoring="accuracy", n_jobs=-1,
    #         train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    #     )
    #
    #     train_scores_mean = np.mean(train_scores, axis=1)
    #     test_scores_mean = np.mean(test_scores, axis=1)
    #
    #     plt.figure()
    #     plt.plot(train_sizes, train_scores_mean, label="Training score")
    #     plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
    #     plt.title(title)
    #     plt.xlabel("Number of training examples")
    #     plt.ylabel("Accuracy")
    #     plt.legend(loc="best")
    #     plt.grid(True)
    #     plt.show()

    def recommend_vehicle_types(self, location: str, date: pd.Timestamp) -> pd.DataFrame:
        """
        Recommend vehicle types for a given location and date using the trained classifier.

        Returns a DataFrame listing each vehicle type with its associated predicted confidence score.
        """
        if not location or not date:
            raise ValueError("Location and date must be provided.")
        if not hasattr(self.vehicle_model, "classes_"):
            raise ValueError("Vehicle recommendation model is not trained.")

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

        X_input, _ = self._prepare_vehicle_data(df_input)
        probs = self.vehicle_model.predict_proba(X_input)

        confidences = []
        for i, row in df_input.iterrows():
            c = row["vehicle_type_encoded"]
            confidences.append(probs[i, c])

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
        Predict seasonal demand for a particular location across multiple vehicle types.

        Steps:
          1. Standardise date column.
          2. Filter for the given location.
          3. Map quarter to season.
          4. Group historical demand and prepare features.
          5. Train a RandomForestRegressor and predict future demand.
        """
        future_seasons = ["Spring", "Summer", "Autumn", "Winter"]

        # Standardise date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        elif "pickup_date" in df.columns:
            df["date"] = pd.to_datetime(df["pickup_date"], dayfirst=True, errors="coerce")
        else:
            raise ValueError("DataFrame must have a 'date' or 'pickup_date' column.")

        # Filter data for the given location
        df_loc = df[df["location"] == location].copy()
        df_loc.dropna(subset=["demand", "vehicle_type", "pickup_date_quarter"], inplace=True)

        quarter_to_season = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
        df_loc["season"] = df_loc["pickup_date_quarter"].map(quarter_to_season)

        grouped = (
            df_loc.groupby(["vehicle_type", "season"], as_index=False)
            .agg({"demand": "mean"})
            .rename(columns={"demand": "avg_historical_demand"})
        )

        # Map season and vehicle type to numeric values for regression
        season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
        grouped["season_num"] = grouped["season"].map(season_map)

        unique_vtypes = grouped["vehicle_type"].unique().tolist()
        vtype_map = {v: i for i, v in enumerate(unique_vtypes)}
        grouped["vehicle_type_num"] = grouped["vehicle_type"].map(vtype_map)

        X_train = grouped[["season_num", "vehicle_type_num"]]
        y_train = grouped["avg_historical_demand"]

        if len(X_train) < 2:
            logging.warning("Not enough data to train a reliable seasonal model.")
            return pd.DataFrame(columns=["season", "location", "vehicle_type", "predicted_demand"])

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Create future data combinations for predictions
        future_combos = [(s, vt) for s in future_seasons for vt in vehicle_types]
        future_df = pd.DataFrame(future_combos, columns=["season", "vehicle_type"])
        future_df["season_num"] = future_df["season"].map(season_map)
        future_df["vehicle_type_num"] = future_df["vehicle_type"].map(vtype_map).fillna(-1)

        X_pred = future_df[["season_num", "vehicle_type_num"]]
        preds = model.predict(X_pred)

        future_df["predicted_demand"] = preds
        future_df["location"] = location
        future_df = future_df[["season", "location", "vehicle_type", "predicted_demand"]]
        future_df.sort_values(["season", "vehicle_type"], inplace=True)

        return future_df

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model and encoders to disk.
        """
        self.vehicle_model.save_model(model_path + "_xgb_model.json")
        joblib.dump({
            "location_encoder": self.location_encoder,
            "vehicle_type_encoder": self.vehicle_type_encoder,
            "feature_cols": self.feature_cols
        }, model_path + "_encoders.pkl")

    def load_model(self, model_path: str) -> None:
        """
        Load the trained model and encoders from disk.
        """
        self.vehicle_model = xgb.XGBClassifier()
        self.vehicle_model.load_model(model_path + "_xgb_model.json")
        data = joblib.load(model_path + "_encoders.pkl")
        self.location_encoder = data["location_encoder"]
        self.vehicle_type_encoder = data["vehicle_type_encoder"]
        self.feature_cols = data["feature_cols"]