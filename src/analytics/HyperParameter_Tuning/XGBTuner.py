import logging
import joblib
import numpy as np
import pandas as pd
import logging
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)  # Add project root to sys.path

# Import the LocationPredictor model correctly
from models.location_predictor import LocationPredictor


class XGBTuner:
    """
    A class responsible for hyperparameter tuning of the XGBoost model.
    """

    def __init__(self):
        self.location_encoder = LabelEncoder()
        self.vehicle_type_encoder = LabelEncoder()

    def tune_model(self, predictor: LocationPredictor, df: pd.DataFrame, cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV with StratifiedKFold.
        The predictor must have its label encoders fitted before calling this method.
        """

        # Ensure date features are present
        df["pickup_date_dayofweek"] = df["pickup_date"].dt.dayofweek
        df["pickup_date_month"] = df["pickup_date"].dt.month
        df["pickup_date_is_weekend"] = df["pickup_date_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        X, y = predictor._prepare_vehicle_data(df)

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [4, 6],
            "learning_rate": [0.01, 0.1],
            "reg_alpha": [0.1, 1],
            "reg_lambda": [1, 1.5],
            "min_child_weight": [1, 3],
            "gamma": [0, 0.1]
        }

        model = xgb.XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=skf,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X, y)

        logging.info("Hyperparameter tuning completed.")
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Accuracy: {grid_search.best_score_:.4f}")

        # Assign the best estimator to the predictor’s vehicle_model
        predictor.vehicle_model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        }


def main():
    # ------------------------------
    # 1. Load Your Real Dataset
    # ------------------------------

    data_path = os.path.join("data", "processed_car_rental.csv")
    df = pd.read_csv(data_path, parse_dates=["pickup_date"])

    # ------------------------------
    # 2. Initialise the Predictor
    # ------------------------------
    predictor = LocationPredictor()

    # ------------------------------
    # 3. Train the Model Initially
    # ------------------------------
    logging.info("Training initial model with default hyperparameters...")
    predictor.fit_encoders(df)
    train_results = predictor.train(df)

    # ----------------------------------------
    # 4. Hyperparameter Tuning in a New Class
    # ----------------------------------------
    tuner = XGBTuner()

    logging.info("Tuning model hyperparameters...")
    tuning_results = tuner.tune_model(predictor, df, cv=5)

    # ------------------------------------------------
    # 5. Retrain & Evaluate Using the Tuned Parameters
    # ------------------------------------------------
    logging.info("Evaluating the tuned model...")
    predictor.train(df)

    # ---------------------------------------------
    # 6. Vehicle Recommendation Example
    # ---------------------------------------------
    sample_location = "London"
    sample_date = pd.Timestamp("2023-05-15")
    recommendations = predictor.recommend_vehicle_types(sample_location, sample_date)
    logging.info("Vehicle Recommendations:\n" + str(recommendations))

    # ---------------------------------------------
    # 7. Seasonal Demand Forecast Example
    # ---------------------------------------------
    sample_vehicle_types = ["Sedan", "SUV"]
    # Ensure your dataset has columns like 'demand', 'pickup_date_quarter', 'location'
    # or adapt as needed
    demand_forecast = predictor.predict_seasonal_demand(df, sample_location, sample_vehicle_types)
    logging.info("Seasonal Demand Predictions:\n" + str(demand_forecast))

    # ---------------------------
    # 8. Save Your Final Model
    # ---------------------------
    predictor.save_model("final_model")
    logging.info("Model and encoders saved to disk.")


if __name__ == "__main__":
    main()
