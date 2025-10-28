"""
Data Loading Module
Handles CSV data ingestion and validation for NexGen Analytics Platform
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import streamlit as st

# Expected columns for each dataset
EXPECTED_COLUMNS = {
    'cost_breakdown': ['Order_ID', 'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                      'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead'],
    'customer_feedback': ['Order_ID', 'Feedback_Date', 'Rating', 'Feedback_Text', 
                         'Would_Recommend', 'Issue_Category'],
    'delivery_performance': ['Order_ID', 'Carrier', 'Promised_Delivery_Days', 'Actual_Delivery_Days',
                           'Delivery_Status', 'Quality_Issue', 'Customer_Rating', 'Delivery_Cost_INR'],
    'orders': ['Order_ID', 'Order_Date', 'Customer_Segment', 'Priority', 'Product_Category',
              'Order_Value_INR', 'Origin', 'Destination', 'Special_Handling'],
    'routes_distance': ['Order_ID', 'Route', 'Distance_KM', 'Fuel_Consumption_L', 
                       'Toll_Charges_INR', 'Traffic_Delay_Minutes', 'Weather_Impact'],
    'vehicle_fleet': ['Vehicle_ID', 'Vehicle_Type', 'Capacity_KG', 'Fuel_Efficiency_KM_per_L',
                     'Current_Location', 'Status', 'Age_Years', 'CO2_Emissions_Kg_per_KM'],
    'warehouse_inventory': ['Warehouse_ID', 'Location', 'Product_Category', 'Current_Stock_Units',
                          'Reorder_Level', 'Storage_Cost_per_Unit', 'Last_Restocked_Date']
}

@st.cache_data
def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all CSV datasets into pandas DataFrames"""
    dataframes = {}
    datasets_path = Path("../datasets")
    
    try:
        # Load each CSV file
        for dataset_name in EXPECTED_COLUMNS.keys():
            file_path = datasets_path / f"{dataset_name}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                dataframes[dataset_name] = df
                st.success(f"✅ Loaded {dataset_name}.csv: {len(df)} rows")
            else:
                st.error(f"❌ File not found: {file_path}")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        
    return dataframes

def validate_data_integrity(dataframes: Dict[str, pd.DataFrame]) -> bool:
    """Validate required columns and data types"""
    validation_passed = True
    
    for dataset_name, df in dataframes.items():
        expected_cols = EXPECTED_COLUMNS.get(dataset_name, [])
        missing_cols = set(expected_cols) - set(df.columns)
        
        if missing_cols:
            st.error(f"❌ {dataset_name}: Missing columns {missing_cols}")
            validation_passed = False
        else:
            st.success(f"✅ {dataset_name}: All required columns present")
    
    return validation_passed

def get_data_summary(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Provide dataset statistics and completeness metrics"""
    summary = {}
    
    for name, df in dataframes.items():
        summary[name] = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'completeness': f"{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%"
        }
    
    return summary