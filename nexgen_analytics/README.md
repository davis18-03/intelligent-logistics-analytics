# NexGen Profit & Risk Intelligence Platform

An interactive analytics dashboard that provides unified visibility into order profitability and delivery risk across NexGen's logistics operations. This platform combines data engineering, machine learning, and interactive visualization to enable data-driven operational decisions.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to project directory**
   ```bash
   cd nexgen_analytics
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure datasets are available**
   - Place CSV files in `../datasets/` directory
   - Required files: cost_breakdown.csv, customer_feedback.csv, delivery_performance.csv, orders.csv, routes_distance.csv, vehicle_fleet.csv, warehouse_inventory.csv

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The dashboard will automatically open at `http://localhost:8501`

## 🎯 Key Features

### 1. **Profit-Risk Quadrant Analysis** (Core Innovation)
- Interactive scatter plot showing profit vs delivery risk for all orders
- Visual quadrants: High Profit/Low Risk (Ideal), High Profit/High Risk (Action Needed), etc.
- Hover details with order information and customer data
- Color-coded by customer segment with size based on order value

### 2. **Machine Learning Risk Prediction**
- RandomForest classifier trained on delivery performance data
- Predicts probability of severe delays for all orders (including new ones)
- Features: Priority, Product Category, Customer Segment, Carrier, Distance, Traffic, Weather
- Cross-validated model with performance metrics

### 3. **Interactive Filtering System**
- Sidebar filters for Customer Segment, Carrier, Product Category, and Priority
- Real-time updates across all visualizations
- Filter statistics and reset functionality

### 4. **Carrier Performance Analytics**
- Average profit and risk score comparison by carrier
- Delivery performance metrics and on-time percentages
- Identification of best/worst performing carriers
- Actionable insights for carrier relationship management

### 5. **Customer Feedback Analysis**
- Rating distribution pie chart and issue category analysis
- Correlation between delivery performance and customer satisfaction
- Identification of orders requiring follow-up (poor ratings)
- Customer satisfaction metrics and trends

### 6. **High-Priority Order Identification**
- Top 10 high-profit, high-risk orders requiring immediate attention
- Priority scoring algorithm combining profit potential and delivery risk
- Actionable recommendations for each priority order
- Executive summary with total value at risk

### 7. **Financial Analysis & Profit Distribution**
- Profit distribution histogram with statistical markers
- Comprehensive financial metrics (mean, median, quartiles)
- Identification of loss-making and low-margin orders
- Correlation analysis with operational factors

### 8. **Report Download & Export** ⭐ NEW
- **Executive Summary Report**: Key metrics and KPIs for leadership
- **Detailed Analytics Export**: Complete filtered dataset with all metrics
- **Priority Orders Report**: Top 10 actionable orders with recommendations
- **Timestamped Files**: Automatic naming with date/time for version control

## 📊 Dashboard Sections

1. **Data Pipeline Status**: Data loading, validation, and processing results
2. **Analytics Dashboard**: Main interactive visualizations and insights
3. **Profit vs Risk Quadrant**: Core 2x2 matrix analysis
4. **Carrier Performance**: Logistics provider comparison and insights
5. **Customer Feedback**: Satisfaction analysis and follow-up identification
6. **Priority Orders**: High-value, at-risk order management
7. **Financial Analysis**: Profit distribution and optimization opportunities
8. **Report Downloads**: Executive summaries and detailed analytics export

## 📁 Project Structure

```
nexgen_analytics/
├── app.py                 # Main Streamlit application with dashboard logic
├── requirements.txt       # Python dependencies (pandas, streamlit, scikit-learn, plotly)
├── data_loader.py        # CSV data ingestion and validation
├── data_processor.py     # Data merging, cost calculation, and profit metrics
├── feature_engineer.py   # ML feature preparation and target variable creation
├── risk_model.py         # RandomForest risk prediction model
├── visualizations.py     # Plotly charts and analysis functions
├── README.md             # This documentation
└── ../datasets/          # CSV data files (external directory)
```

## 🔧 Technical Implementation

### Data Pipeline
- **Loading**: Automated CSV ingestion with validation
- **Processing**: Left joins preserving all 200 orders
- **Metrics**: Profit calculation (Order Value - Total Costs)
- **Features**: ML-ready feature engineering with missing value handling

### Machine Learning
- **Algorithm**: RandomForestClassifier with hyperparameter tuning
- **Preprocessing**: OneHotEncoder for categoricals, StandardScaler for numericals
- **Training**: Cross-validated on ~150 completed orders
- **Prediction**: Risk scores for all 200 orders (including incomplete)

### Visualization
- **Framework**: Plotly for interactive charts
- **Quadrants**: Dynamic profit/risk thresholds based on median values
- **Filtering**: Real-time updates across all components
- **Responsiveness**: Optimized for various screen sizes

## 📈 Business Impact & Value

### Cost Reduction
- **Low Profit, High Risk Orders**: Identify and re-evaluate problematic segments
- **Carrier Optimization**: Data-driven carrier selection and negotiation
- **Resource Allocation**: Focus on high-value, manageable orders

### Customer Experience Improvement
- **Proactive Management**: Early intervention for high-value, at-risk orders
- **Service Quality**: Correlation analysis between delivery and satisfaction
- **Follow-up Actions**: Systematic identification of dissatisfied customers

### Data-Driven Culture
- **Unified View**: Single platform for operational decision-making
- **Predictive Operations**: Move from reactive to proactive management
- **Actionable Insights**: Clear recommendations for each business scenario

## 🚀 Future Enhancements

- Real-time data integration and automated updates
- Advanced ML models (XGBoost, Neural Networks)
- Predictive analytics for demand forecasting
- Mobile-responsive design and notifications
- Integration with existing ERP/CRM systems

## 📞 Support

For technical support or feature requests, please refer to the project documentation or contact the development team.

---

**Built with**: Python, Streamlit, Pandas, Scikit-learn, Plotly
**Status**: Production Ready MVP
**Last Updated**: October 2025