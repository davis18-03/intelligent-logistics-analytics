"""
Risk Prediction Model
Machine learning pipeline for delivery delay risk scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple
import streamlit as st

class RiskModel:
    """Risk prediction model for delivery delays"""
    
    def __init__(self):
        self.model_pipeline = None
        self.feature_names = None
        self.categorical_features = ['Priority', 'Product_Category', 'Customer_Segment', 'Carrier', 'Weather_Impact']
        self.numerical_features = ['Distance_KM', 'Traffic_Delay_Minutes']
        self.is_trained = False
    
    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline for features"""
        
        # Categorical preprocessing
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # Numerical preprocessing  
        numerical_transformer = StandardScaler()
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return preprocessor
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the risk prediction model"""
        
        if len(X) < 10:
            raise ValueError("Insufficient training data. Need at least 10 samples.")
        
        # Create preprocessing pipeline
        preprocessor = self._create_preprocessing_pipeline()
        
        # Create full pipeline with model
        self.model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        # Train the model
        self.model_pipeline.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model_pipeline, X, y, cv=min(5, len(X)//2), scoring='accuracy')
        
        # Calculate training metrics
        train_accuracy = self.model_pipeline.score(X, y)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'training_samples': len(X),
            'positive_class_ratio': y.mean()
        }
        
        return metrics
    
    def predict_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return risk probability scores for all orders"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle missing values by filling with defaults
        X_processed = X.copy()
        
        # Fill missing categorical features
        for feature in self.categorical_features:
            if feature in X_processed.columns:
                X_processed[feature] = X_processed[feature].fillna('Unknown')
        
        # Fill missing numerical features with median from training
        for feature in self.numerical_features:
            if feature in X_processed.columns:
                median_val = X_processed[feature].median()
                X_processed[feature] = X_processed[feature].fillna(median_val)
        
        # Predict probabilities (return probability of class 1 - severely delayed)
        try:
            risk_probabilities = self.model_pipeline.predict_proba(X_processed)[:, 1]
        except Exception as e:
            # Fallback: return default risk scores if prediction fails
            st.warning(f"Prediction failed for some orders: {str(e)}")
            risk_probabilities = np.full(len(X_processed), 0.5)  # Default medium risk
        
        return risk_probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance rankings"""
        
        if not self.is_trained:
            return {}
        
        # Get feature importance from the trained model
        classifier = self.model_pipeline.named_steps['classifier']
        
        # Get feature names after preprocessing
        preprocessor = self.model_pipeline.named_steps['preprocessor']
        
        # Get feature names for numerical features
        num_feature_names = self.numerical_features
        
        # Get feature names for categorical features (after one-hot encoding)
        cat_feature_names = []
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features).tolist()
        
        # Combine all feature names
        all_feature_names = num_feature_names + cat_feature_names
        
        # Get importance scores
        importance_scores = classifier.feature_importances_
        
        # Create feature importance dictionary
        feature_importance = dict(zip(all_feature_names, importance_scores))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance

def train_risk_model(df: pd.DataFrame, features: list) -> Tuple[RiskModel, Dict]:
    """Train risk model and return model with metrics"""
    
    # Filter to complete training data
    training_data = df.dropna(subset=features + ['Is_Severely_Delayed'])
    
    if len(training_data) == 0:
        raise ValueError("No complete training data available")
    
    # Prepare features and target
    X = training_data[features]
    y = training_data['Is_Severely_Delayed']
    
    # Initialize and train model
    model = RiskModel()
    metrics = model.train(X, y)
    
    return model, metrics