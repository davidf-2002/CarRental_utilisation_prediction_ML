import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from typing import Tuple, Dict, List
import logging
from datetime import timedelta

class LocationPredictor:
    def __init__(self):
        self.demand_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.vehicle_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        self.location_encoder = LabelEncoder()
        self.vehicle_type_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare features for prediction"""
        try:
            # Create temporal features from date
            df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['pickup_date'])
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
            df['season'] = pd.cut(df['month'], 
                                bins=[0, 3, 6, 9, 12], 
                                labels=['Winter', 'Spring', 'Summer', 'Fall'])
            
            # Calculate rental duration if needed
            if 'rental_duration' not in df.columns and 'dropoff_date' in df.columns:
                df['rental_duration'] = (pd.to_datetime(df['dropoff_date']) - df['date']).dt.total_seconds() / (24 * 3600)
                df['rental_duration'] = df['rental_duration'].fillna(1)
            
            # Ensure required columns exist
            required_cols = ['pickUp.city', 'rate.daily', 'rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Encode categorical variables
            df['location_encoded'] = self.location_encoder.fit_transform(df['pickUp.city'])
            
            # Always include vehicle_type_encoded, using a default if not present
            if 'vehicle.type' in df.columns:
                df['vehicle_type_encoded'] = self.vehicle_type_encoder.transform(df['vehicle.type'])
            else:
                # If no vehicle type provided, use a neutral encoding (mean of all types)
                n_classes = len(self.vehicle_type_encoder.classes_)
                df['vehicle_type_encoded'] = np.mean(range(n_classes))
            
            # Calculate location metrics
            agg_dict = {
                'rate.daily': ['mean', 'std'],
                'rating': 'mean'
            }
            if 'rental_duration' in df.columns:
                agg_dict['rental_duration'] = ['mean', 'std']
            
            metrics = df.groupby('pickUp.city').agg(agg_dict).fillna(0)
            
            # Flatten column names
            metrics.columns = ['avg_duration', 'std_duration', 'avg_rate', 'std_rate', 'avg_rating'] \
                if 'rental_duration' in df.columns else ['avg_rate', 'std_rate', 'avg_rating']
            metrics = metrics.reset_index()
            
            # Merge metrics back
            df = df.merge(metrics, on='pickUp.city', how='left')
            
            # Select features
            base_features = [
                'month', 'day_of_week', 'is_weekend',
                'avg_rate', 'std_rate', 'avg_rating',
                'vehicle_type_encoded'  # Always include vehicle type
            ]
            
            if 'rental_duration' in df.columns:
                base_features.extend(['avg_duration', 'std_duration'])
            
            X = df[base_features].fillna(0)
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=base_features)
            
            feature_info = {
                'location_mapping': dict(zip(self.location_encoder.classes_, self.location_encoder.transform(self.location_encoder.classes_))),
                'vehicle_type_mapping': dict(zip(self.vehicle_type_encoder.classes_, self.vehicle_type_encoder.transform(self.vehicle_type_encoder.classes_))),
                'features': base_features
            }
            
            return X_scaled, feature_info
            
        except Exception as e:
            raise ValueError(f"Error preparing features: {str(e)}")
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train both demand prediction and vehicle type recommendation models"""
        try:
            # Validate required columns
            required_cols = ['pickUp.city', 'vehicle.type', 'pickup_date', 'rate.daily', 'rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for training: {', '.join(missing_cols)}")
            
            # Ensure date columns are datetime
            df['date'] = pd.to_datetime(df['pickup_date'])
            if 'dropoff_date' in df.columns:
                df['dropoff_date'] = pd.to_datetime(df['dropoff_date'])
            
            # First fit the vehicle type encoder so it's available for feature preparation
            self.vehicle_type_encoder.fit(df['vehicle.type'].unique())
            
            # Calculate demand per location and vehicle type
            demand = df.groupby(['pickUp.city', 'vehicle.type', 'date']).size().reset_index(name='demand')
            merged_data = demand.merge(df, on=['pickUp.city', 'vehicle.type', 'date'], how='left')
            
            # Train demand prediction model
            X, feature_info = self.prepare_features(merged_data)
            y_demand = merged_data['demand']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_demand, test_size=0.2, random_state=42)
            self.demand_model.fit(X_train, y_train)
            
            # Calculate demand model metrics
            y_pred = self.demand_model.predict(X_test)
            self.feature_importance = dict(zip(X.columns, self.demand_model.feature_importances_))
            
            demand_metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'feature_importance': self.feature_importance
            }
            
            # Train vehicle type recommendation model
            df_vehicle = df.copy()
            df_vehicle['date'] = df_vehicle['pickup_date']  # Ensure date column exists
            X_vehicle, _ = self.prepare_features(df_vehicle)
            y_vehicle = self.vehicle_type_encoder.transform(df_vehicle['vehicle.type'])  # Already fitted above
            
            X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vehicle, y_vehicle, test_size=0.2, random_state=42)
            self.vehicle_model.fit(X_train_v, y_train_v)
            
            # Calculate vehicle model metrics
            y_pred_v = self.vehicle_model.predict(X_test_v)
            vehicle_metrics = {
                'accuracy': accuracy_score(y_test_v, y_pred_v),
                'classification_report': classification_report(y_test_v, y_pred_v, 
                    target_names=self.vehicle_type_encoder.classes_)
            }
            
            return {
                'demand_metrics': demand_metrics,
                'vehicle_metrics': vehicle_metrics
            }
            
        except Exception as e:
            raise ValueError(f"Error training models: {str(e)}")
    
    def predict_future_demand(self, future_dates: List[pd.Timestamp], locations: List[str], vehicle_types: List[str], historical_averages: pd.DataFrame) -> pd.DataFrame:
        """Predict demand for given locations and vehicle types"""
        try:
            # Validate inputs
            if not future_dates or not locations or not vehicle_types:
                raise ValueError("Must provide at least one date, location, and vehicle type")
            
            # Create combinations
            combinations = [(d, l, v) for d in future_dates for l in locations for v in vehicle_types]
            pred_data = pd.DataFrame(combinations, columns=['date', 'pickUp.city', 'vehicle.type'])


            # Add required columns with historical averages
            # Merge historical data for the relevant combination of month, location, and vehicle type
            pred_data['month'] = pred_data['date'].dt.month

            # Merge the historical averages with the prediction data
            pred_data = pred_data.merge(historical_averages, on=['pickUp.city', 'vehicle.type', 'month'], how='left')

            # Ensure that we have filled all columns
            pred_data['dropoff_date'] = pred_data['date'] + pd.Timedelta(days=pred_data['rental_duration'])
            pred_data['rental_duration'] = pred_data['rental_duration'].fillna(
                1)  # Use historical average or default to 1 if NaN
            pred_data['rate.daily'] = pred_data['rate.daily'].fillna(
                100)  # Use historical average or default to 100 if NaN
            pred_data['rating'] = pred_data['rating'].fillna(4.0)  # Use historical average or default to 4.0 if NaN


            # Get predictions
            X_pred, _ = self.prepare_features(pred_data)
            predictions = np.maximum(0, self.demand_model.predict(X_pred))  # Ensure non-negative predictions
            
            # Create results dataframe with proper formatting
            results = pd.DataFrame({
                'date': pd.to_datetime(pred_data['date']),
                'location': pred_data['pickUp.city'],
                'vehicle_type': pred_data['vehicle.type'],
                'predicted_demand': predictions.round(1)
            })
            
            # Sort by date and location for better readability
            results = results.sort_values(['date', 'location', 'vehicle_type'])
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error predicting demand: {str(e)}")
    
    def recommend_vehicle_types(self, location: str, date: pd.Timestamp) -> pd.DataFrame:
        """Recommend vehicle types for a given location and date"""
        try:
            # Validate inputs
            if not location or not date:
                raise ValueError("Location and date must be provided")
            
            if not isinstance(date, pd.Timestamp):
                date = pd.to_datetime(date)
            
            # Ensure model is trained
            if not hasattr(self.vehicle_model, 'classes_'):
                raise ValueError("Vehicle recommendation model is not trained")
            
            # Create input data for each vehicle type to get features
            vehicle_types = self.vehicle_type_encoder.classes_
            input_data = pd.DataFrame([
                {
                    'date': date,
                    'pickUp.city': location,
                    'dropoff_date': date + pd.Timedelta(days=1),
                    'rate.daily': 100,  # Default daily rate
                    'rating': 4.0,  # Default rating
                    'vehicle.type': vtype  # Add vehicle type
                } for vtype in vehicle_types
            ])
            
            # Get features
            X_pred, _ = self.prepare_features(input_data)
            
            # Get probabilities for each vehicle type
            probs = self.vehicle_model.predict_proba(X_pred[:1])  # Only need first row since all rows have same features except vehicle type
            
            # Create recommendations dataframe with proper formatting
            recommendations = pd.DataFrame({
                'vehicle_type': vehicle_types,
                'confidence': probs[0]
            })
            
            # Sort by confidence and format percentages
            recommendations = recommendations.sort_values('confidence', ascending=False)
            recommendations['confidence'] = recommendations['confidence'].round(3)
            
            return recommendations
            
        except Exception as e:
            raise ValueError(f"Error recommending vehicle types: {str(e)}")
    
    def save_model(self, path: str):
        """Save both models and their components"""
        joblib.dump({
            'demand_model': self.demand_model,
            'vehicle_model': self.vehicle_model,
            'location_encoder': self.location_encoder,
            'vehicle_type_encoder': self.vehicle_type_encoder,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, path)
    
    def load_model(self, path: str):
        """Load both models and their components"""
        components = joblib.load(path)
        self.demand_model = components['demand_model']
        self.vehicle_model = components['vehicle_model']
        self.location_encoder = components['location_encoder']
        self.vehicle_type_encoder = components['vehicle_type_encoder']
        self.scaler = components['scaler']
        self.feature_importance = components['feature_importance']
