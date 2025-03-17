import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the car rental dataset
    """
    return pd.read_csv(file_path)

def extract_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Extract various datetime features from a datetime column
    """
    # Convert to datetime if not already
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract basic time components
    df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
    df[f'{datetime_col}_day'] = df[datetime_col].dt.day
    df[f'{datetime_col}_month'] = df[datetime_col].dt.month
    df[f'{datetime_col}_year'] = df[datetime_col].dt.year
    df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
    df[f'{datetime_col}_quarter'] = df[datetime_col].dt.quarter
    
    # Create period of day
    df[f'{datetime_col}_period'] = pd.cut(
        df[f'{datetime_col}_hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    
    # Is weekend flag
    df[f'{datetime_col}_is_weekend'] = df[f'{datetime_col}_dayofweek'].isin([5, 6]).astype(int)
    
    return df

def handle_missing_values(df: pd.DataFrame, numeric_strategy: str = 'median', 
                         categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    """
    df = df.copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if categorical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif categorical_strategy == 'unknown':
                df[col].fillna('Unknown', inplace=True)
    
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate entries from the dataset
    """
    return df.drop_duplicates()

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Handle outliers in specified numeric columns
    """
    df = df.copy()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def encode_categorical_variables(df: pd.DataFrame, columns: List[str], 
                               method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical variables using specified method
    """
    df = df.copy()
    encoders = {}
    
    if method == 'label':
        for col in columns:
            encoder = LabelEncoder()
            df[f'{col}_encoded'] = encoder.fit_transform(df[col])
            encoders[col] = encoder
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=columns, prefix=columns)
    
    return df, encoders

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self, df):
        """Basic data cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna()  # For now, we'll drop NA values
        
        return df
    
    def prepare_features(self, df, target_column):
        """Prepare features for model training"""
        df = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        # Encode categorical variables
        df_encoded, self.encoders = encode_categorical_variables(df, categorical_cols, method='label')
        
        # Separate features and target
        if target_column in df.select_dtypes(include=['object']):
            # If target is categorical, encode it
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(df[target_column])
            self.encoders[target_column] = target_encoder
        else:
            y = df[target_column]
        
        # Select features (encoded categorical and numeric)
        feature_cols = [col for col in df_encoded.columns if col != target_column and 'encoded' in col] + list(numeric_cols)
        X = df_encoded[feature_cols]
        
        # Scale numeric features
        X_scaled = X.copy()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
