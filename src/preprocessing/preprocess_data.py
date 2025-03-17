import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing_utils import (
    load_data,
    extract_datetime_features,
    handle_missing_values,
    remove_duplicates,
    handle_outliers,
    encode_categorical_variables
)

def main():
    # Set up paths
    current_dir = Path(__file__).parent.parent.parent
    input_file = current_dir / 'data' / 'NewCarRental.csv'
    output_file = current_dir / 'data' / 'processed_car_rental.csv'
    
    # Load the data
    print("Loading data...")
    df = load_data(str(input_file))
    
    # Basic info about the dataset
    print("\nDataset Info:")
    print(f"Shape before preprocessing: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    df = remove_duplicates(df)
    
    # Handle missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    
    # Extract datetime features
    print("\nExtracting datetime features...")
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    if not datetime_columns.empty:
        for col in datetime_columns:
            df = extract_datetime_features(df, col)
    else:
        # Try to identify datetime columns that might be stored as objects
        potential_datetime_cols = df.select_dtypes(include=['object']).columns
        for col in potential_datetime_cols:
            try:
                # Check if column can be converted to datetime
                pd.to_datetime(df[col])
                print(f"Converting {col} to datetime and extracting features...")
                df = extract_datetime_features(df, col)
            except:
                continue
    
    # Handle outliers for numeric columns
    print("\nHandling outliers...")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df = handle_outliers(df, columns=numeric_cols)
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = encode_categorical_variables(df, columns=categorical_cols)
    
    # Save processed dataset
    print("\nSaving processed dataset...")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")
    
    # Final dataset info
    print("\nFinal dataset shape:", df.shape)
    print("\nColumns in processed dataset:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()
