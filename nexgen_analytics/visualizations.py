"""
Visualization Components
Individual chart and graph generation for dashboard
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple
import numpy as np

def create_profit_risk_scatter(df: pd.DataFrame) -> go.Figure:
    """Create main quadrant analysis scatter plot"""
    
    # Filter to orders with both profit and risk score data
    plot_data = df.dropna(subset=['Profit', 'Risk_Score']).copy()
    
    if len(plot_data) == 0:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for profit-risk analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Calculate quadrant lines (median values)
    risk_median = plot_data['Risk_Score'].median()
    profit_median = plot_data['Profit'].median()
    
    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x='Risk_Score',
        y='Profit',
        color='Customer_Segment',
        size='Order_Value_INR',
        hover_data={
            'Order_ID': True,
            'Customer_Segment': True,
            'Order_Value_INR': ':,.0f',
            'Carrier': True,
            'Product_Category': True,
            'Risk_Score': ':.3f',
            'Profit': ':,.0f'
        },
        title="Profit vs Risk Analysis - Quadrant View",
        labels={
            'Risk_Score': 'Risk Score (Probability of Severe Delay)',
            'Profit': 'Profit (₹)',
            'Customer_Segment': 'Customer Segment'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(
        y=profit_median, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"Median Profit: ₹{profit_median:.0f}"
    )
    
    fig.add_vline(
        x=risk_median, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"Median Risk: {risk_median:.3f}"
    )
    
    # Add quadrant labels
    y_range = plot_data['Profit'].max() - plot_data['Profit'].min()
    x_range = plot_data['Risk_Score'].max() - plot_data['Risk_Score'].min()
    
    # High Profit, Low Risk (Top Left)
    fig.add_annotation(
        x=risk_median - 0.1 * x_range,
        y=profit_median + 0.4 * y_range,
        text="🟢 High Profit<br>Low Risk<br>(Ideal)",
        showarrow=False,
        bgcolor="rgba(0,255,0,0.1)",
        bordercolor="green",
        font=dict(size=10, color="green")
    )
    
    # High Profit, High Risk (Top Right)
    fig.add_annotation(
        x=risk_median + 0.1 * x_range,
        y=profit_median + 0.4 * y_range,
        text="🟡 High Profit<br>High Risk<br>(Action Needed)",
        showarrow=False,
        bgcolor="rgba(255,255,0,0.1)",
        bordercolor="orange",
        font=dict(size=10, color="orange")
    )
    
    # Low Profit, Low Risk (Bottom Left)
    fig.add_annotation(
        x=risk_median - 0.1 * x_range,
        y=profit_median - 0.4 * y_range,
        text="🔵 Low Profit<br>Low Risk<br>(Optimize)",
        showarrow=False,
        bgcolor="rgba(0,0,255,0.1)",
        bordercolor="blue",
        font=dict(size=10, color="blue")
    )
    
    # Low Profit, High Risk (Bottom Right)
    fig.add_annotation(
        x=risk_median + 0.1 * x_range,
        y=profit_median - 0.4 * y_range,
        text="🔴 Low Profit<br>High Risk<br>(Critical)",
        showarrow=False,
        bgcolor="rgba(255,0,0,0.1)",
        bordercolor="red",
        font=dict(size=10, color="red")
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(title_text="Profit (₹)")
    
    return fig

def get_quadrant_analysis(df: pd.DataFrame) -> dict:
    """Analyze orders by quadrant and return statistics"""
    
    # Filter to orders with both profit and risk score data
    analysis_data = df.dropna(subset=['Profit', 'Risk_Score']).copy()
    
    if len(analysis_data) == 0:
        return {}
    
    # Calculate quadrant thresholds
    risk_median = analysis_data['Risk_Score'].median()
    profit_median = analysis_data['Profit'].median()
    
    # Categorize orders into quadrants
    analysis_data['Quadrant'] = 'Unknown'
    
    # High Profit, Low Risk
    mask = (analysis_data['Profit'] >= profit_median) & (analysis_data['Risk_Score'] < risk_median)
    analysis_data.loc[mask, 'Quadrant'] = 'High Profit, Low Risk'
    
    # High Profit, High Risk
    mask = (analysis_data['Profit'] >= profit_median) & (analysis_data['Risk_Score'] >= risk_median)
    analysis_data.loc[mask, 'Quadrant'] = 'High Profit, High Risk'
    
    # Low Profit, Low Risk
    mask = (analysis_data['Profit'] < profit_median) & (analysis_data['Risk_Score'] < risk_median)
    analysis_data.loc[mask, 'Quadrant'] = 'Low Profit, Low Risk'
    
    # Low Profit, High Risk
    mask = (analysis_data['Profit'] < profit_median) & (analysis_data['Risk_Score'] >= risk_median)
    analysis_data.loc[mask, 'Quadrant'] = 'Low Profit, High Risk'
    
    # Calculate statistics for each quadrant
    quadrant_stats = {}
    for quadrant in analysis_data['Quadrant'].unique():
        if quadrant != 'Unknown':
            quadrant_data = analysis_data[analysis_data['Quadrant'] == quadrant]
            quadrant_stats[quadrant] = {
                'count': len(quadrant_data),
                'avg_profit': quadrant_data['Profit'].mean(),
                'avg_risk': quadrant_data['Risk_Score'].mean(),
                'total_value': quadrant_data['Order_Value_INR'].sum(),
                'orders': quadrant_data['Order_ID'].tolist()
            }
    
    return quadrant_stats

def create_carrier_analytics(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Create carrier performance charts"""
    
    # Filter to orders with carrier data
    carrier_data = df.dropna(subset=['Carrier']).copy()
    
    if len(carrier_data) == 0:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No carrier data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return empty_fig, empty_fig
    
    # Calculate carrier metrics
    carrier_metrics = carrier_data.groupby('Carrier').agg({
        'Profit': ['mean', 'sum', 'count'],
        'Risk_Score': 'mean',
        'Order_Value_INR': ['mean', 'sum'],
        'Delivery_Status': lambda x: (x == 'On-Time').sum() / len(x) * 100 if len(x) > 0 else 0
    }).round(2)
    
    # Flatten column names
    carrier_metrics.columns = ['Avg_Profit', 'Total_Profit', 'Order_Count', 
                              'Avg_Risk_Score', 'Avg_Order_Value', 'Total_Order_Value', 
                              'OnTime_Delivery_Pct']
    carrier_metrics = carrier_metrics.reset_index()
    
    # Create profit by carrier chart
    profit_fig = px.bar(
        carrier_metrics,
        x='Carrier',
        y='Avg_Profit',
        title='Average Profit by Carrier',
        labels={'Avg_Profit': 'Average Profit (₹)', 'Carrier': 'Carrier'},
        color='Avg_Profit',
        color_continuous_scale='RdYlGn'
    )
    
    # Add order count as text on bars
    profit_fig.update_traces(
        text=carrier_metrics['Order_Count'],
        texttemplate='Orders: %{text}',
        textposition='outside'
    )
    
    profit_fig.update_layout(
        showlegend=False,
        height=400
    )
    
    # Create risk score by carrier chart
    risk_fig = px.bar(
        carrier_metrics,
        x='Carrier',
        y='Avg_Risk_Score',
        title='Average Risk Score by Carrier',
        labels={'Avg_Risk_Score': 'Average Risk Score', 'Carrier': 'Carrier'},
        color='Avg_Risk_Score',
        color_continuous_scale='RdYlBu_r'  # Reverse scale so red = high risk
    )
    
    # Add on-time delivery percentage as text
    risk_fig.update_traces(
        text=carrier_metrics['OnTime_Delivery_Pct'],
        texttemplate='On-Time: %{text:.1f}%',
        textposition='outside'
    )
    
    risk_fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return profit_fig, risk_fig

def get_carrier_performance_summary(df: pd.DataFrame) -> dict:
    """Get detailed carrier performance statistics"""
    
    carrier_data = df.dropna(subset=['Carrier']).copy()
    
    if len(carrier_data) == 0:
        return {}
    
    # Calculate comprehensive carrier metrics
    carrier_summary = {}
    
    for carrier in carrier_data['Carrier'].unique():
        carrier_orders = carrier_data[carrier_data['Carrier'] == carrier]
        
        # Basic metrics
        total_orders = len(carrier_orders)
        avg_profit = carrier_orders['Profit'].mean() if 'Profit' in carrier_orders.columns else 0
        avg_risk = carrier_orders['Risk_Score'].mean() if 'Risk_Score' in carrier_orders.columns else 0
        
        # Delivery performance
        delivery_data = carrier_orders.dropna(subset=['Delivery_Status'])
        if len(delivery_data) > 0:
            on_time_pct = (delivery_data['Delivery_Status'] == 'On-Time').sum() / len(delivery_data) * 100
            severely_delayed_pct = (delivery_data['Delivery_Status'] == 'Severely-Delayed').sum() / len(delivery_data) * 100
        else:
            on_time_pct = 0
            severely_delayed_pct = 0
        
        # Customer satisfaction
        rating_data = carrier_orders.dropna(subset=['Customer_Rating'])
        avg_rating = rating_data['Customer_Rating'].mean() if len(rating_data) > 0 else 0
        
        carrier_summary[carrier] = {
            'total_orders': total_orders,
            'avg_profit': avg_profit,
            'avg_risk_score': avg_risk,
            'on_time_delivery_pct': on_time_pct,
            'severely_delayed_pct': severely_delayed_pct,
            'avg_customer_rating': avg_rating,
            'total_revenue': carrier_orders['Order_Value_INR'].sum()
        }
    
    return carrier_summary

def create_customer_feedback_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Create customer feedback analysis visualizations"""
    
    # Filter to orders with feedback data
    feedback_data = df.dropna(subset=['Rating']).copy()
    
    if len(feedback_data) == 0:
        # Return empty figure if no data
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No customer feedback data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return empty_fig, empty_fig
    
    # Create rating distribution pie chart
    rating_counts = feedback_data['Rating'].value_counts().sort_index()
    
    # Define colors for ratings (1=red, 5=green)
    colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#44aa44']
    
    pie_fig = go.Figure(data=[go.Pie(
        labels=[f"{rating} Star{'s' if rating != 1 else ''}" for rating in rating_counts.index],
        values=rating_counts.values,
        hole=0.3,
        marker_colors=[colors[int(rating)-1] for rating in rating_counts.index]
    )])
    
    pie_fig.update_layout(
        title="Customer Rating Distribution",
        height=400,
        showlegend=True
    )
    
    # Create issue category analysis
    issue_data = feedback_data.dropna(subset=['Issue_Category'])
    
    if len(issue_data) > 0:
        issue_counts = issue_data['Issue_Category'].value_counts()
        
        issue_fig = px.bar(
            x=issue_counts.values,
            y=issue_counts.index,
            orientation='h',
            title='Issues by Category',
            labels={'x': 'Number of Issues', 'y': 'Issue Category'},
            color=issue_counts.values,
            color_continuous_scale='Reds'
        )
        
        issue_fig.update_layout(
            height=400,
            showlegend=False
        )
    else:
        issue_fig = go.Figure()
        issue_fig.add_annotation(
            text="No issue category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    return pie_fig, issue_fig

def get_customer_feedback_insights(df: pd.DataFrame) -> dict:
    """Analyze customer feedback and return insights"""
    
    feedback_data = df.dropna(subset=['Rating']).copy()
    
    if len(feedback_data) == 0:
        return {}
    
    insights = {}
    
    # Overall satisfaction metrics
    insights['total_feedback'] = len(feedback_data)
    insights['avg_rating'] = feedback_data['Rating'].mean()
    insights['rating_distribution'] = feedback_data['Rating'].value_counts().to_dict()
    
    # Satisfaction levels
    insights['satisfied_customers'] = (feedback_data['Rating'] >= 4).sum()
    insights['dissatisfied_customers'] = (feedback_data['Rating'] <= 2).sum()
    insights['satisfaction_rate'] = insights['satisfied_customers'] / len(feedback_data) * 100
    
    # Issue analysis
    issue_data = feedback_data.dropna(subset=['Issue_Category'])
    if len(issue_data) > 0:
        insights['top_issues'] = issue_data['Issue_Category'].value_counts().head(3).to_dict()
    else:
        insights['top_issues'] = {}
    
    # Correlation with delivery performance
    delivery_feedback = feedback_data.dropna(subset=['Delivery_Status'])
    if len(delivery_feedback) > 0:
        delivery_rating_corr = delivery_feedback.groupby('Delivery_Status')['Rating'].mean().to_dict()
        insights['rating_by_delivery_status'] = delivery_rating_corr
    
    # Negative feedback orders for follow-up
    negative_feedback = feedback_data[feedback_data['Rating'] <= 2]
    insights['negative_feedback_orders'] = negative_feedback['Order_ID'].tolist()
    
    return insights

def create_profit_distribution(df: pd.DataFrame) -> go.Figure:
    """Create profit distribution histogram"""
    
    # Filter to orders with profit data
    profit_data = df.dropna(subset=['Profit']).copy()
    
    if len(profit_data) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No profit data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Create histogram
    fig = px.histogram(
        profit_data,
        x='Profit',
        nbins=30,
        title='Profit Distribution Across All Orders',
        labels={'Profit': 'Profit (₹)', 'count': 'Number of Orders'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add statistical lines
    mean_profit = profit_data['Profit'].mean()
    median_profit = profit_data['Profit'].median()
    
    fig.add_vline(
        x=mean_profit,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ₹{mean_profit:.0f}"
    )
    
    fig.add_vline(
        x=median_profit,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: ₹{median_profit:.0f}"
    )
    
    # Add zero profit line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="black",
        annotation_text="Break-even"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def get_financial_analysis(df: pd.DataFrame) -> dict:
    """Perform comprehensive financial analysis"""
    
    profit_data = df.dropna(subset=['Profit']).copy()
    
    if len(profit_data) == 0:
        return {}
    
    analysis = {}
    
    # Basic profit statistics
    analysis['total_orders'] = len(profit_data)
    analysis['total_profit'] = profit_data['Profit'].sum()
    analysis['avg_profit'] = profit_data['Profit'].mean()
    analysis['median_profit'] = profit_data['Profit'].median()
    analysis['profit_std'] = profit_data['Profit'].std()
    
    # Profit quartiles
    analysis['q1_profit'] = profit_data['Profit'].quantile(0.25)
    analysis['q3_profit'] = profit_data['Profit'].quantile(0.75)
    
    # Profitability analysis
    analysis['profitable_orders'] = (profit_data['Profit'] > 0).sum()
    analysis['loss_making_orders'] = (profit_data['Profit'] < 0).sum()
    analysis['profitability_rate'] = analysis['profitable_orders'] / len(profit_data) * 100
    
    # Low margin orders (bottom 10%)
    low_margin_threshold = profit_data['Profit'].quantile(0.1)
    analysis['low_margin_orders'] = (profit_data['Profit'] <= low_margin_threshold).sum()
    analysis['low_margin_threshold'] = low_margin_threshold
    
    # High margin orders (top 10%)
    high_margin_threshold = profit_data['Profit'].quantile(0.9)
    analysis['high_margin_orders'] = (profit_data['Profit'] >= high_margin_threshold).sum()
    analysis['high_margin_threshold'] = high_margin_threshold
    
    # Profit by operational factors
    if 'Distance_KM' in profit_data.columns:
        # Correlation with distance
        distance_data = profit_data.dropna(subset=['Distance_KM'])
        if len(distance_data) > 0:
            analysis['profit_distance_correlation'] = distance_data['Profit'].corr(distance_data['Distance_KM'])
    
    if 'Carrier' in profit_data.columns:
        # Profit by carrier
        carrier_profit = profit_data.groupby('Carrier')['Profit'].agg(['mean', 'sum', 'count']).to_dict()
        analysis['profit_by_carrier'] = carrier_profit
    
    # Identify problematic orders
    analysis['negative_profit_orders'] = profit_data[profit_data['Profit'] < 0]['Order_ID'].tolist()
    analysis['lowest_profit_orders'] = profit_data.nsmallest(5, 'Profit')[['Order_ID', 'Profit']].to_dict('records')
    
    return analysis