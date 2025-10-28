"""
Feature Engineering Module
Prepares data for machine learning model training
"""

import pandas as pd
from typing import Tuple, List
import numpy as np

# Define the features to use for ML model
MODEL_FEATURES = [
    'Priority',
    'Product_Category', 
    'Customer_Segment',
    'Carrier',
    'Distance_KM',
    'Traffic_Delay_Minutes',
    'Weather_Impact'
]

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Add Is_Severely_Delayed column based on delivery status"""
    df_with_target = df.copy()
    
    # Create binary target: 1 if severely delayed, 0 otherwise, NaN if no delivery data
    df_with_target['Is_Severely_Delayed'] = df_with_target['Delivery_Status'].map({
        'Severely-Delayed': 1,
        'Slightly-Delayed': 0,
        'On-Time': 0
    })
    
    return df_with_target

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean features for ML model"""
    df_features = df.copy()
    
    # Handle missing values in categorical features
    categorical_features = ['Priority', 'Product_Category', 'Customer_Segment', 'Carrier', 'Weather_Impact']
    for feature in categorical_features:
        if feature in df_features.columns:
            df_features[feature] = df_features[feature].fillna('Unknown')
    
    # Handle missing values in numerical features
    numerical_features = ['Distance_KM', 'Traffic_Delay_Minutes']
    for feature in numerical_features:
        if feature in df_features.columns:
            # Fill with median for numerical features
            median_value = df_features[feature].median()
            df_features[feature] = df_features[feature].fillna(median_value)
    
    # Ensure Weather_Impact has consistent categories
    if 'Weather_Impact' in df_features.columns:
        # Replace None/NaN with 'None' string for consistency
        df_features['Weather_Impact'] = df_features['Weather_Impact'].fillna('None')
        df_features['Weather_Impact'] = df_features['Weather_Impact'].replace('None', 'None')
    
    return df_features

def split_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate complete vs incomplete orders for training"""
    
    # Complete orders have all required features and target variable
    required_cols = MODEL_FEATURES + ['Is_Severely_Delayed']
    
    # Check which orders have complete data
    complete_mask = df[required_cols].notna().all(axis=1)
    
    df_complete = df[complete_mask].copy()
    df_incomplete = df[~complete_mask].copy()
    
    return df_complete, df_incomplete

def validate_feature_distributions(df: pd.DataFrame) -> dict:
    """Validate feature distributions and detect potential issues"""
    validation_results = {}
    
    for feature in MODEL_FEATURES:
        if feature in df.columns:
            if df[feature].dtype == 'object':  # Categorical
                value_counts = df[feature].value_counts()
                validation_results[feature] = {
                    'type': 'categorical',
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'missing_count': df[feature].isna().sum()
                }
            else:  # Numerical
                validation_results[feature] = {
                    'type': 'numerical',
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'missing_count': df[feature].isna().sum()
                }
    
    return validation_results

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return processed DataFrame and feature list"""
    
    # Create target variable
    df_with_target = create_target_variable(df)
    
    # Prepare features
    df_prepared = prepare_features(df_with_target)
    
    # Split into complete and incomplete data
    df_complete, df_incomplete = split_training_data(df_prepared)
    
    # Combine back for return (maintaining all orders)
    df_final = pd.concat([df_complete, df_incomplete], ignore_index=True)
    
    return df_final, MODEL_FEATURES

def get_feature_engineering_stats(df: pd.DataFrame) -> dict:
    """Get statistics about feature engineering results"""
    
    # Split data to get training statistics
    df_complete, df_incomplete = split_training_data(df)
    
    stats = {
        'total_orders': len(df),
        'complete_orders': len(df_complete),
        'incomplete_orders': len(df_incomplete),
        'training_data_pct': round(len(df_complete) / len(df) * 100, 1) if len(df) > 0 else 0,
        'target_distribution': df['Is_Severely_Delayed'].value_counts().to_dict() if 'Is_Severely_Delayed' in df.columns else {}
    }
    
    return stats