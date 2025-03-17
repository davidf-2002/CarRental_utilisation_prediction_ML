# Location Demand Prediction Model Documentation

## Overview
The Location Demand Prediction system is designed to forecast optimal vehicle placement across different locations based on historical rental data. It uses machine learning to predict future demand patterns and provide actionable insights for fleet management.

## Model Architecture

### 1. Feature Engineering
The model uses the following features:

#### Temporal Features
- Month (1-12)
- Day of week (0-6)
- Weekend flag (0/1)

#### Location-based Features
- Average rental duration per location
- Standard deviation of rental duration
- Average daily rate
- Standard deviation of daily rate
- Average customer rating

#### Vehicle Features
- Vehicle type (encoded)
- Location (encoded)

### 2. Model Selection
- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - n_estimators: 100
  - max_depth: 15
  - random_state: 42

### 3. Data Processing
1. **Categorical Encoding**:
   - Location names → Label encoded
   - Vehicle types → Label encoded

2. **Feature Scaling**:
   - StandardScaler applied to numerical features
   - Preserves relative importance of features

3. **Target Variable**:
   - Daily demand aggregation per location and vehicle type

## Dashboard Features

### 1. Model Training Section
- Dataset overview
- Feature importance visualization
- Model performance metrics:
  - R² Score
  - Root Mean Square Error (RMSE)

### 2. Prediction Analysis
- **Monthly Demand Heatmap**:
  - X-axis: Months
  - Y-axis: Vehicle Types
  - Color intensity: Predicted demand

- **Location Comparison**:
  - Line plot comparing demand across locations
  - Filtered by vehicle type
  - 12-month forecast

### 3. Recommendations
- Top 3 vehicle types per location
- Peak season identification
- Monthly demand forecasts

## Usage Instructions

1. **Data Upload**:
   - Upload CSV file containing rental history
   - Required columns:
     - pickup_date
     - pickUp.city
     - vehicle.type
     - rental_duration
     - rate.daily
     - rating

2. **Model Training**:
   - Click "Train Model" button
   - Review performance metrics
   - Analyze feature importance

3. **Demand Analysis**:
   - Select location from dropdown
   - View demand heatmap
   - Compare locations for specific vehicle types

4. **Interpreting Results**:
   - Heatmap colors indicate demand intensity
   - Line plots show temporal trends
   - Recommendations provide actionable insights

## Technical Details

### Model Performance
- Cross-validation: 80/20 train-test split
- Metrics:
  - R² Score: Indicates prediction accuracy
  - RMSE: Shows prediction error magnitude

### Prediction Process
1. Generate future dates (12 months)
2. Create feature combinations
3. Apply feature engineering
4. Scale features
5. Generate predictions
6. Post-process results

### Model Persistence
- Model saved as 'location_predictor.joblib'
- Includes:
  - Trained model
  - Label encoders
  - Feature scaler
  - Feature importance scores

## Limitations and Considerations
1. Predictions assume historical patterns continue
2. Weather and special events not considered
3. Economic factors not included
4. Limited to existing locations and vehicle types

## Future Improvements
1. Include weather data
2. Add economic indicators
3. Incorporate competitor information
4. Implement automated retraining
5. Add confidence intervals for predictions
