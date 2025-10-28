"""
Data Processing Module
Handles data transformation, merging, and metric calculations
"""

import pandas as pd
from typing import Dict
import streamlit as st

def calculate_total_costs(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total costs by summing all cost components"""
    cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                   'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
    
    # Create a copy to avoid modifying original
    df_costs = cost_df.copy()
    
    # Calculate total cost
    df_costs['Total_Cost'] = df_costs[cost_columns].sum(axis=1)
    
    # Return only Order_ID and Total_Cost
    return df_costs[['Order_ID', 'Total_Cost']]

def merge_datasets(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create master dataset through left joins on Order_ID"""
    
    # Start with orders as the base (ensures all 200 orders are preserved)
    df_master = dataframes['orders'].copy()
    
    # Calculate costs first
    if 'cost_breakdown' in dataframes:
        df_costs = calculate_total_costs(dataframes['cost_breakdown'])
        df_master = df_master.merge(df_costs, on='Order_ID', how='left')
    
    # Merge delivery performance data
    if 'delivery_performance' in dataframes:
        df_master = df_master.merge(dataframes['delivery_performance'], on='Order_ID', how='left')
    
    # Merge route distance data
    if 'routes_distance' in dataframes:
        route_cols = ['Order_ID', 'Distance_KM', 'Traffic_Delay_Minutes', 'Weather_Impact']
        df_routes = dataframes['routes_distance'][route_cols]
        df_master = df_master.merge(df_routes, on='Order_ID', how='left')
    
    # Merge customer feedback data
    if 'customer_feedback' in dataframes:
        feedback_cols = ['Order_ID', 'Rating', 'Issue_Category']
        df_feedback = dataframes['customer_feedback'][feedback_cols]
        df_master = df_master.merge(df_feedback, on='Order_ID', how='left')
    
    return df_master

def calculate_profit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add profit calculations to dataset"""
    df_with_profit = df.copy()
    
    # Calculate Profit = Order_Value_INR - Total_Cost
    df_with_profit['Profit'] = df_with_profit['Order_Value_INR'] - df_with_profit['Total_Cost']
    
    # Calculate profit margin percentage
    df_with_profit['Profit_Margin_Pct'] = (
        df_with_profit['Profit'] / df_with_profit['Order_Value_INR'] * 100
    ).round(2)
    
    return df_with_profit

def create_master_dataset(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create consolidated dataset with all metrics"""
    
    # Merge all datasets
    df_master = merge_datasets(dataframes)
    
    # Calculate profit metrics
    df_master = calculate_profit_metrics(df_master)
    
    return df_master

def get_data_completeness_stats(df: pd.DataFrame) -> Dict:
    """Get statistics about data completeness"""
    total_orders = len(df)
    
    stats = {
        'total_orders': total_orders,
        'orders_with_costs': df['Total_Cost'].notna().sum(),
        'orders_with_delivery_data': df['Delivery_Status'].notna().sum(),
        'orders_with_routes': df['Distance_KM'].notna().sum(),
        'orders_with_feedback': df['Rating'].notna().sum(),
        'complete_orders': df[['Total_Cost', 'Delivery_Status', 'Distance_KM']].notna().all(axis=1).sum()
    }
    
    # Calculate percentages
    for key in list(stats.keys()):
        if key != 'total_orders':
            stats[f'{key}_pct'] = round(stats[key] / total_orders * 100, 1)
    
    return stats