"""
NexGen Profit & Risk Intelligence Platform
Main Streamlit application for logistics analytics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Configure Streamlit page
st.set_page_config(
    page_title="NexGen Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_sidebar_filters(df: pd.DataFrame) -> Dict:
    """Create sidebar filters and return filter values"""
    st.sidebar.header("🔍 Filters")
    
    filters = {}
    
    # Customer Segment filter
    customer_segments = ['All'] + sorted(df['Customer_Segment'].dropna().unique().tolist())
    filters['customer_segment'] = st.sidebar.selectbox(
        "Customer Segment", 
        customer_segments,
        index=0
    )
    
    # Carrier filter
    carriers = ['All'] + sorted(df['Carrier'].dropna().unique().tolist())
    filters['carrier'] = st.sidebar.selectbox(
        "Carrier",
        carriers, 
        index=0
    )
    
    # Product Category filter
    categories = ['All'] + sorted(df['Product_Category'].dropna().unique().tolist())
    filters['product_category'] = st.sidebar.selectbox(
        "Product Category",
        categories,
        index=0
    )
    
    # Priority filter
    priorities = ['All'] + sorted(df['Priority'].dropna().unique().tolist())
    filters['priority'] = st.sidebar.selectbox(
        "Priority",
        priorities,
        index=0
    )
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Apply customer segment filter
    if filters['customer_segment'] != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == filters['customer_segment']]
    
    # Apply carrier filter
    if filters['carrier'] != 'All':
        filtered_df = filtered_df[filtered_df['Carrier'] == filters['carrier']]
    
    # Apply product category filter
    if filters['product_category'] != 'All':
        filtered_df = filtered_df[filtered_df['Product_Category'] == filters['product_category']]
    
    # Apply priority filter
    if filters['priority'] != 'All':
        filtered_df = filtered_df[filtered_df['Priority'] == filters['priority']]
    
    return filtered_df

def display_filter_stats(original_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Display statistics about filtered data"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Filter Results")
    
    original_count = len(original_df)
    filtered_count = len(filtered_df)
    
    st.sidebar.metric("Filtered Orders", f"{filtered_count:,}")
    st.sidebar.metric("Percentage", f"{(filtered_count/original_count*100):.1f}%")
    
    # Show profit statistics for filtered data
    profit_data = filtered_df.dropna(subset=['Profit'])
    if len(profit_data) > 0:
        avg_profit = profit_data['Profit'].mean()
        st.sidebar.metric("Avg Profit (Filtered)", f"₹{avg_profit:.0f}")

def create_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create executive summary report"""
    
    summary_data = []
    
    # Overall metrics
    total_orders = len(df)
    profit_orders = df.dropna(subset=['Profit'])
    
    if len(profit_orders) > 0:
        total_profit = profit_orders['Profit'].sum()
        avg_profit = profit_orders['Profit'].mean()
        profitable_orders = (profit_orders['Profit'] > 0).sum()
        profitability_rate = profitable_orders / len(profit_orders) * 100
    else:
        total_profit = avg_profit = profitable_orders = profitability_rate = 0
    
    # Risk metrics
    risk_orders = df.dropna(subset=['Risk_Score'])
    if len(risk_orders) > 0:
        avg_risk = risk_orders['Risk_Score'].mean()
        high_risk_orders = (risk_orders['Risk_Score'] > 0.7).sum()
    else:
        avg_risk = high_risk_orders = 0
    
    # Create summary
    summary_data.append({
        'Metric': 'Total Orders',
        'Value': total_orders,
        'Description': 'Total number of orders in filtered dataset'
    })
    
    summary_data.append({
        'Metric': 'Total Profit',
        'Value': f"{total_profit:,.0f}",
        'Description': 'Sum of all order profits (INR)'
    })
    
    summary_data.append({
        'Metric': 'Average Profit per Order',
        'Value': f"{avg_profit:.0f}",
        'Description': 'Mean profit across all orders (INR)'
    })
    
    summary_data.append({
        'Metric': 'Profitability Rate',
        'Value': f"{profitability_rate:.1f}%",
        'Description': 'Percentage of orders with positive profit'
    })
    
    summary_data.append({
        'Metric': 'Average Risk Score',
        'Value': f"{avg_risk:.3f}",
        'Description': 'Mean risk score (0-1, higher = more risk)'
    })
    
    summary_data.append({
        'Metric': 'High Risk Orders',
        'Value': high_risk_orders,
        'Description': 'Orders with risk score > 0.7'
    })
    
    return pd.DataFrame(summary_data)

def main():
    """Main application entry point"""
    st.title("🚚 NexGen Profit & Risk Intelligence Platform")
    st.markdown("---")
    
    # Import data loading functions
    from data_loader import load_all_data, validate_data_integrity, get_data_summary
    
    # Data loading section
    st.header("📊 Data Pipeline Status")
    
    with st.spinner("Loading datasets..."):
        dataframes = load_all_data()
    
    if dataframes:
        st.subheader("Data Validation")
        validation_passed = validate_data_integrity(dataframes)
        
        if validation_passed:
            st.success("🎉 All datasets loaded and validated successfully!")
            
            # Import data processing functions
            from data_processor import create_master_dataset, get_data_completeness_stats
            
            # Create master dataset
            with st.spinner("Processing and merging datasets..."):
                df_master = create_master_dataset(dataframes)
            
            st.success("✅ Master dataset created successfully!")
            
            # Display data completeness statistics
            st.subheader("Data Processing Results")
            completeness_stats = get_data_completeness_stats(df_master)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", completeness_stats['total_orders'])
            
            with col2:
                st.metric("Orders with Costs", 
                         f"{completeness_stats['orders_with_costs']} ({completeness_stats['orders_with_costs_pct']}%)")
            
            with col3:
                st.metric("Orders with Delivery Data", 
                         f"{completeness_stats['orders_with_delivery_data']} ({completeness_stats['orders_with_delivery_data_pct']}%)")
            
            with col4:
                st.metric("Complete Orders", 
                         f"{completeness_stats['complete_orders']} ({completeness_stats['complete_orders_pct']}%)")
            
            # Display sample of master dataset
            st.subheader("Master Dataset Preview")
            st.dataframe(df_master.head(10), use_container_width=True)
            
            # Feature Engineering
            from feature_engineer import engineer_features, get_feature_engineering_stats
            
            with st.spinner("Engineering features for ML model..."):
                df_engineered, feature_list = engineer_features(df_master)
            
            st.success("✅ Feature engineering completed!")
            
            # Display feature engineering statistics
            st.subheader("Feature Engineering Results")
            fe_stats = get_feature_engineering_stats(df_engineered)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Orders", 
                         f"{fe_stats['complete_orders']} ({fe_stats['training_data_pct']}%)")
            
            with col2:
                st.metric("Incomplete Orders", fe_stats['incomplete_orders'])
            
            with col3:
                if fe_stats['target_distribution']:
                    severely_delayed = fe_stats['target_distribution'].get(1, 0)
                    st.metric("Severely Delayed", severely_delayed)
            
            # Show profit statistics for completed orders
            completed_orders = df_engineered.dropna(subset=['Profit'])
            if len(completed_orders) > 0:
                st.subheader("Profit Analysis Preview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_profit = completed_orders['Profit'].mean()
                    st.metric("Average Profit", f"₹{avg_profit:.2f}")
                
                with col2:
                    total_profit = completed_orders['Profit'].sum()
                    st.metric("Total Profit", f"₹{total_profit:,.2f}")
                
                with col3:
                    avg_margin = completed_orders['Profit_Margin_Pct'].mean()
                    st.metric("Average Margin", f"{avg_margin:.1f}%")
            
            # Train Risk Model
            from risk_model import train_risk_model
            
            try:
                with st.spinner("Training risk prediction model..."):
                    risk_model, model_metrics = train_risk_model(df_engineered, feature_list)
                
                st.success("✅ Risk model trained successfully!")
                
                # Display model performance metrics
                st.subheader("Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Accuracy", f"{model_metrics['train_accuracy']:.3f}")
                
                with col2:
                    st.metric("CV Accuracy", f"{model_metrics['cv_mean_accuracy']:.3f} ± {model_metrics['cv_std_accuracy']:.3f}")
                
                with col3:
                    st.metric("Training Samples", model_metrics['training_samples'])
                
                # Generate risk scores for all orders
                with st.spinner("Generating risk scores..."):
                    df_engineered['Risk_Score'] = risk_model.predict_risk_scores(df_engineered[feature_list])
                
                st.success("✅ Risk scores generated for all orders!")
                
                # Display risk score statistics
                st.subheader("Risk Score Distribution")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_risk = df_engineered['Risk_Score'].mean()
                    st.metric("Average Risk Score", f"{avg_risk:.3f}")
                
                with col2:
                    high_risk_count = (df_engineered['Risk_Score'] > 0.7).sum()
                    st.metric("High Risk Orders (>0.7)", high_risk_count)
                
                with col3:
                    low_risk_count = (df_engineered['Risk_Score'] < 0.3).sum()
                    st.metric("Low Risk Orders (<0.3)", low_risk_count)
                
                # Display feature importance
                feature_importance = risk_model.get_feature_importance()
                if feature_importance:
                    st.subheader("Feature Importance")
                    
                    # Show top 5 most important features
                    top_features = list(feature_importance.items())[:5]
                    for feature, importance in top_features:
                        st.write(f"**{feature}**: {importance:.3f}")
                
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.info("Continuing with basic analytics...")
                # Set default risk scores
                df_engineered['Risk_Score'] = 0.5
            
            # Setup sidebar filters
            filters = setup_sidebar_filters(df_engineered)
            
            # Apply filters to data
            df_filtered = apply_filters(df_engineered, filters)
            
            # Display filter statistics
            display_filter_stats(df_engineered, df_filtered)
            
            # Main Dashboard Content
            st.header("📈 Analytics Dashboard")
            
            # Show filtered data summary
            st.subheader("Filtered Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(df_filtered))
            
            with col2:
                profit_orders = df_filtered.dropna(subset=['Profit'])
                if len(profit_orders) > 0:
                    avg_profit = profit_orders['Profit'].mean()
                    st.metric("Average Profit", f"₹{avg_profit:.0f}")
                else:
                    st.metric("Average Profit", "N/A")
            
            with col3:
                if 'Risk_Score' in df_filtered.columns:
                    avg_risk = df_filtered['Risk_Score'].mean()
                    st.metric("Average Risk Score", f"{avg_risk:.3f}")
                else:
                    st.metric("Average Risk Score", "N/A")
            
            with col4:
                high_risk_orders = (df_filtered['Risk_Score'] > 0.7).sum() if 'Risk_Score' in df_filtered.columns else 0
                st.metric("High Risk Orders", high_risk_orders)
            
            # Profit-Risk Quadrant Analysis (Main Innovation)
            st.subheader("🎯 Profit vs Risk Quadrant Analysis")
            
            from visualizations import create_profit_risk_scatter, get_quadrant_analysis
            
            # Create and display the main quadrant plot
            quadrant_fig = create_profit_risk_scatter(df_filtered)
            st.plotly_chart(quadrant_fig, use_container_width=True)
            
            # Display quadrant analysis statistics
            quadrant_stats = get_quadrant_analysis(df_filtered)
            
            if quadrant_stats:
                st.subheader("📊 Quadrant Analysis Summary")
                
                # Create columns for each quadrant
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                quadrant_colors = {
                    'High Profit, Low Risk': '🟢',
                    'High Profit, High Risk': '🟡', 
                    'Low Profit, Low Risk': '🔵',
                    'Low Profit, High Risk': '🔴'
                }
                
                columns = [col1, col2, col3, col4]
                quadrant_names = list(quadrant_stats.keys())
                
                for i, (quadrant, stats) in enumerate(quadrant_stats.items()):
                    if i < len(columns):
                        with columns[i]:
                            color_icon = quadrant_colors.get(quadrant, '⚪')
                            st.markdown(f"**{color_icon} {quadrant}**")
                            st.metric("Orders", stats['count'])
                            st.metric("Avg Profit", f"₹{stats['avg_profit']:.0f}")
                            st.metric("Avg Risk", f"{stats['avg_risk']:.3f}")
                            st.metric("Total Value", f"₹{stats['total_value']:,.0f}")
            
            # Carrier Performance Analytics
            st.subheader("🚛 Carrier Performance Analytics")
            
            from visualizations import create_carrier_analytics, get_carrier_performance_summary
            
            # Create carrier performance charts
            profit_chart, risk_chart = create_carrier_analytics(df_filtered)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(profit_chart, use_container_width=True)
            
            with col2:
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Display carrier performance summary
            carrier_summary = get_carrier_performance_summary(df_filtered)
            
            if carrier_summary:
                st.subheader("📋 Carrier Performance Summary")
                
                # Create a summary table
                summary_data = []
                for carrier, metrics in carrier_summary.items():
                    summary_data.append({
                        'Carrier': carrier,
                        'Total Orders': metrics['total_orders'],
                        'Avg Profit (₹)': f"{metrics['avg_profit']:.0f}",
                        'Avg Risk Score': f"{metrics['avg_risk_score']:.3f}",
                        'On-Time %': f"{metrics['on_time_delivery_pct']:.1f}%",
                        'Severely Delayed %': f"{metrics['severely_delayed_pct']:.1f}%",
                        'Avg Rating': f"{metrics['avg_customer_rating']:.1f}",
                        'Total Revenue (₹)': f"{metrics['total_revenue']:,.0f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Highlight best and worst performers
                if len(carrier_summary) > 1:
                    st.subheader("🏆 Carrier Insights")
                    
                    # Best profit performer
                    best_profit_carrier = max(carrier_summary.items(), key=lambda x: x[1]['avg_profit'])
                    st.success(f"**Best Profit Performance**: {best_profit_carrier[0]} (₹{best_profit_carrier[1]['avg_profit']:.0f} avg profit)")
                    
                    # Lowest risk carrier
                    lowest_risk_carrier = min(carrier_summary.items(), key=lambda x: x[1]['avg_risk_score'])
                    st.info(f"**Lowest Risk**: {lowest_risk_carrier[0]} ({lowest_risk_carrier[1]['avg_risk_score']:.3f} avg risk score)")
                    
                    # Highest risk carrier
                    highest_risk_carrier = max(carrier_summary.items(), key=lambda x: x[1]['avg_risk_score'])
                    if highest_risk_carrier[1]['avg_risk_score'] > 0.6:
                        st.warning(f"**Attention Needed**: {highest_risk_carrier[0]} has high risk score ({highest_risk_carrier[1]['avg_risk_score']:.3f})")
            
            # Customer Feedback Analysis
            st.subheader("💬 Customer Feedback Analysis")
            
            from visualizations import create_customer_feedback_charts, get_customer_feedback_insights
            
            # Create feedback charts
            rating_chart, issue_chart = create_customer_feedback_charts(df_filtered)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(rating_chart, use_container_width=True)
            
            with col2:
                st.plotly_chart(issue_chart, use_container_width=True)
            
            # Display feedback insights
            feedback_insights = get_customer_feedback_insights(df_filtered)
            
            if feedback_insights:
                st.subheader("📊 Customer Satisfaction Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Feedback", feedback_insights['total_feedback'])
                
                with col2:
                    st.metric("Average Rating", f"{feedback_insights['avg_rating']:.1f}/5")
                
                with col3:
                    st.metric("Satisfaction Rate", f"{feedback_insights['satisfaction_rate']:.1f}%")
                
                with col4:
                    st.metric("Dissatisfied Customers", feedback_insights['dissatisfied_customers'])
                
                # Show top issues if available
                if feedback_insights['top_issues']:
                    st.subheader("🔍 Top Customer Issues")
                    for issue, count in feedback_insights['top_issues'].items():
                        st.write(f"• **{issue}**: {count} reports")
                
                # Show rating by delivery status
                if 'rating_by_delivery_status' in feedback_insights:
                    st.subheader("📦 Rating by Delivery Performance")
                    for status, rating in feedback_insights['rating_by_delivery_status'].items():
                        color = "🟢" if rating >= 4 else "🟡" if rating >= 3 else "🔴"
                        st.write(f"{color} **{status}**: {rating:.1f}/5 average rating")
                
                # Alert for negative feedback orders
                if feedback_insights['negative_feedback_orders']:
                    st.subheader("⚠️ Orders Requiring Follow-up")
                    st.warning(f"Found {len(feedback_insights['negative_feedback_orders'])} orders with poor ratings (≤2 stars)")
                    
                    if len(feedback_insights['negative_feedback_orders']) <= 10:
                        st.write("Order IDs:", ", ".join(feedback_insights['negative_feedback_orders']))
                    else:
                        st.write(f"First 10 Order IDs: {', '.join(feedback_insights['negative_feedback_orders'][:10])}")
            
            # High-Priority Order Identification
            st.subheader("🚨 High-Priority Orders Requiring Attention")
            
            # Identify high-profit, high-risk orders
            priority_data = df_filtered.dropna(subset=['Profit', 'Risk_Score']).copy()
            
            if len(priority_data) > 0:
                # Calculate priority score (combination of profit and risk)
                # Normalize profit and risk to 0-1 scale for fair combination
                profit_normalized = (priority_data['Profit'] - priority_data['Profit'].min()) / (priority_data['Profit'].max() - priority_data['Profit'].min())
                risk_normalized = priority_data['Risk_Score']
                
                # Priority score: high profit + high risk = high priority
                priority_data['Priority_Score'] = profit_normalized * 0.6 + risk_normalized * 0.4
                
                # Get top 10 high-priority orders
                top_priority = priority_data.nlargest(10, 'Priority_Score')
                
                # Display priority orders table
                priority_display = top_priority[[
                    'Order_ID', 'Customer_Segment', 'Product_Category', 'Carrier',
                    'Order_Value_INR', 'Profit', 'Risk_Score', 'Priority_Score', 'Delivery_Status'
                ]].copy()
                
                # Format columns for better display
                priority_display['Order_Value_INR'] = priority_display['Order_Value_INR'].apply(lambda x: f"₹{x:,.0f}")
                priority_display['Profit'] = priority_display['Profit'].apply(lambda x: f"₹{x:.0f}")
                priority_display['Risk_Score'] = priority_display['Risk_Score'].apply(lambda x: f"{x:.3f}")
                priority_display['Priority_Score'] = priority_display['Priority_Score'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(priority_display, use_container_width=True)
                
                # Provide actionable insights for top 3 orders
                st.subheader("💡 Actionable Insights")
                
                for i, (_, order) in enumerate(top_priority.head(3).iterrows()):
                    with st.expander(f"🔥 Priority #{i+1}: Order {order['Order_ID']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Customer**: {order['Customer_Segment']}")
                            st.write(f"**Product**: {order['Product_Category']}")
                            st.write(f"**Carrier**: {order['Carrier']}")
                            st.write(f"**Order Value**: ₹{order['Order_Value_INR']:,.0f}")
                        
                        with col2:
                            st.write(f"**Profit**: ₹{order['Profit']:.0f}")
                            st.write(f"**Risk Score**: {order['Risk_Score']:.3f}")
                            st.write(f"**Status**: {order['Delivery_Status']}")
                            st.write(f"**Priority Score**: {order['Priority_Score']:.3f}")
                        
                        # Provide specific recommendations
                        recommendations = []
                        
                        if order['Risk_Score'] > 0.7:
                            recommendations.append("🚨 **High delay risk** - Consider expedited handling or alternative carrier")
                        
                        if order['Profit'] > priority_data['Profit'].quantile(0.8):
                            recommendations.append("💰 **High-value customer** - Ensure premium service and proactive communication")
                        
                        if order['Customer_Segment'] == 'Enterprise':
                            recommendations.append("🏢 **Enterprise client** - Assign dedicated account manager")
                        
                        if recommendations:
                            st.write("**Recommended Actions:**")
                            for rec in recommendations:
                                st.write(rec)
                
                # Summary statistics
                st.subheader("📈 Priority Order Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_priority_profit = top_priority['Profit'].mean()
                    st.metric("Avg Profit (Top 10)", f"₹{avg_priority_profit:.0f}")
                
                with col2:
                    avg_priority_risk = top_priority['Risk_Score'].mean()
                    st.metric("Avg Risk Score (Top 10)", f"{avg_priority_risk:.3f}")
                
                with col3:
                    total_priority_value = top_priority['Order_Value_INR'].sum()
                    st.metric("Total Value at Risk", f"₹{total_priority_value:,.0f}")
                
            else:
                st.warning("No orders with complete profit and risk data available for priority analysis.")
            
            # Financial Analysis and Profit Distribution
            st.subheader("💰 Financial Analysis & Profit Distribution")
            
            from visualizations import create_profit_distribution, get_financial_analysis
            
            # Create profit distribution chart
            profit_dist_chart = create_profit_distribution(df_filtered)
            st.plotly_chart(profit_dist_chart, use_container_width=True)
            
            # Display financial analysis
            financial_analysis = get_financial_analysis(df_filtered)
            
            if financial_analysis:
                st.subheader("📊 Financial Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Profit", f"₹{financial_analysis['total_profit']:,.0f}")
                    st.metric("Average Profit", f"₹{financial_analysis['avg_profit']:.0f}")
                
                with col2:
                    st.metric("Median Profit", f"₹{financial_analysis['median_profit']:.0f}")
                    st.metric("Profit Std Dev", f"₹{financial_analysis['profit_std']:.0f}")
                
                with col3:
                    st.metric("Profitable Orders", f"{financial_analysis['profitable_orders']}")
                    st.metric("Profitability Rate", f"{financial_analysis['profitability_rate']:.1f}%")
                
                with col4:
                    st.metric("Loss-Making Orders", f"{financial_analysis['loss_making_orders']}")
                    st.metric("Low Margin Orders", f"{financial_analysis['low_margin_orders']}")
                
                # Profit quartiles
                st.subheader("📈 Profit Distribution Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Q1 (25th percentile)", f"₹{financial_analysis['q1_profit']:.0f}")
                
                with col2:
                    st.metric("Q2 (Median)", f"₹{financial_analysis['median_profit']:.0f}")
                
                with col3:
                    st.metric("Q3 (75th percentile)", f"₹{financial_analysis['q3_profit']:.0f}")
                
                # Alerts for problematic orders
                if financial_analysis['negative_profit_orders']:
                    st.subheader("⚠️ Orders Requiring Cost Optimization")
                    st.error(f"Found {len(financial_analysis['negative_profit_orders'])} orders with negative profit")
                    
                    if len(financial_analysis['negative_profit_orders']) <= 10:
                        st.write("Order IDs with losses:", ", ".join(financial_analysis['negative_profit_orders']))
                
                # Show lowest profit orders for investigation
                if financial_analysis['lowest_profit_orders']:
                    st.subheader("🔍 Lowest Profit Orders (Investigation Needed)")
                    
                    lowest_profit_df = pd.DataFrame(financial_analysis['lowest_profit_orders'])
                    lowest_profit_df['Profit'] = lowest_profit_df['Profit'].apply(lambda x: f"₹{x:.0f}")
                    st.dataframe(lowest_profit_df, use_container_width=True)
                
                # Operational insights
                if 'profit_distance_correlation' in financial_analysis:
                    st.subheader("🚛 Operational Insights")
                    corr = financial_analysis['profit_distance_correlation']
                    if abs(corr) > 0.3:
                        direction = "negatively" if corr < 0 else "positively"
                        st.info(f"Profit is {direction} correlated with distance (r={corr:.3f})")
                    else:
                        st.info(f"Weak correlation between profit and distance (r={corr:.3f})")
            
            # Define display columns for reports
            display_columns = ['Order_ID', 'Customer_Segment', 'Priority', 'Product_Category', 
                             'Order_Value_INR', 'Profit', 'Risk_Score', 'Delivery_Status']
            available_columns = [col for col in display_columns if col in df_filtered.columns]
            
            # Report Download Section
            st.subheader("📄 Download Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Executive Summary CSV
                if st.button("📊 Download Executive Summary"):
                    exec_summary = create_executive_summary(df_filtered)
                    csv_data = exec_summary.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="📥 Executive Summary (CSV)",
                        data=csv_data,
                        file_name=f"nexgen_executive_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Detailed Analytics CSV
                if st.button("📈 Download Detailed Analytics"):
                    detailed_data = df_filtered[available_columns].copy() if len(df_filtered) > 0 else pd.DataFrame()
                    if not detailed_data.empty:
                        # Clean data for CSV export - remove currency symbols
                        if 'Order_Value_INR' in detailed_data.columns:
                            detailed_data['Order_Value_INR'] = detailed_data['Order_Value_INR'].round(2)
                        if 'Profit' in detailed_data.columns:
                            detailed_data['Profit'] = detailed_data['Profit'].round(2)
                        if 'Risk_Score' in detailed_data.columns:
                            detailed_data['Risk_Score'] = detailed_data['Risk_Score'].round(4)
                        
                        csv_data = detailed_data.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📥 Detailed Analytics (CSV)",
                            data=csv_data,
                            file_name=f"nexgen_detailed_analytics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
            
            with col3:
                # Priority Orders Report
                if st.button("🚨 Download Priority Orders"):
                    priority_data = df_filtered.dropna(subset=['Profit', 'Risk_Score']).copy()
                    if len(priority_data) > 0:
                        # Calculate priority score
                        profit_normalized = (priority_data['Profit'] - priority_data['Profit'].min()) / (priority_data['Profit'].max() - priority_data['Profit'].min())
                        priority_data['Priority_Score'] = profit_normalized * 0.6 + priority_data['Risk_Score'] * 0.4
                        top_priority = priority_data.nlargest(10, 'Priority_Score')
                        
                        # Clean numerical data for export
                        if 'Order_Value_INR' in top_priority.columns:
                            top_priority['Order_Value_INR'] = top_priority['Order_Value_INR'].round(2)
                        if 'Profit' in top_priority.columns:
                            top_priority['Profit'] = top_priority['Profit'].round(2)
                        if 'Risk_Score' in top_priority.columns:
                            top_priority['Risk_Score'] = top_priority['Risk_Score'].round(4)
                        if 'Priority_Score' in top_priority.columns:
                            top_priority['Priority_Score'] = top_priority['Priority_Score'].round(4)
                        
                        csv_data = top_priority.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📥 Priority Orders (CSV)",
                            data=csv_data,
                            file_name=f"nexgen_priority_orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
            
            # Display sample of filtered data
            st.subheader("Filtered Dataset Preview")
            
            if len(df_filtered) > 0:
                st.dataframe(df_filtered[available_columns].head(10), use_container_width=True)
            else:
                st.warning("No orders match the selected filters.")
            
            # Display feature list
            st.subheader("ML Model Features")
            st.write("Features selected for risk prediction model:")
            for i, feature in enumerate(feature_list, 1):
                st.write(f"{i}. **{feature}**")
            
            # Display dataset summary
            st.subheader("Dataset Summary")
            summary = get_data_summary(dataframes)
            summary_df = pd.DataFrame(summary).T
            st.dataframe(summary_df, use_container_width=True)
            
        else:
            st.error("❌ Data validation failed. Please check the datasets.")
    else:
        st.error("❌ No datasets loaded. Please check the datasets folder.")
    
    # Display project structure info
    with st.expander("📁 Project Structure"):
        st.code("""
        nexgen_analytics/
        ├── app.py                 # Main Streamlit application
        ├── requirements.txt       # Python dependencies
        ├── data_loader.py        # Data loading and validation
        ├── data_processor.py     # Data processing and metrics
        ├── feature_engineer.py   # ML feature engineering
        ├── risk_model.py         # Risk prediction model
        ├── visualizations.py     # Chart and graph components
        └── datasets/             # CSV data files
        """)

if __name__ == "__main__":
    main()