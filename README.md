# 🚚 Intelligent Logistics Analytics Platform

An enterprise-grade business intelligence platform that transforms logistics operations through predictive analytics and strategic insights. Built with Python, Streamlit, and Machine Learning to enable data-driven decision making in supply chain management.

![Platform Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-red)
![ML](https://img.shields.io/badge/ML-RandomForest-orange)

## 🎯 **Key Features**

### 🔥 **Profit-Risk Quadrant Analysis** (Core Innovation)
- Interactive 2x2 matrix visualization showing profit vs delivery risk
- Strategic quadrants: Ideal, Action Needed, Optimize, Critical
- Executive decision-making framework for operational prioritization

### 🤖 **ML-Powered Risk Prediction**
- RandomForest classifier predicting delivery delay probability
- 90%+ accuracy on logistics performance data
- Real-time risk scoring for all orders including new shipments

### 📊 **Comprehensive Business Intelligence**
- **Carrier Performance Analytics**: Compare logistics providers
- **Customer Feedback Analysis**: Satisfaction tracking and follow-up identification
- **Financial Optimization**: Profit distribution and cost analysis
- **Priority Order Management**: Top 10 high-value, at-risk shipments

### 📄 **Executive Reporting**
- Downloadable CSV reports for stakeholder communication
- Executive summary with key business metrics
- Detailed analytics export for further analysis
- Priority orders report with actionable recommendations

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/davis18-03/intelligent-logistics-analytics.git
   cd intelligent-logistics-analytics
   ```

2. **Set up virtual environment**
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

4. **Prepare your data**
   - Place CSV files in `datasets/` directory
   - Required files: `cost_breakdown.csv`, `customer_feedback.csv`, `delivery_performance.csv`, `orders.csv`, `routes_distance.csv`, `vehicle_fleet.csv`, `warehouse_inventory.csv`

5. **Run the application**
   ```bash
   cd nexgen_analytics
   streamlit run app.py
   ```

6. **Access the platform**
   - Open your browser to `http://localhost:8501`
   - Use sidebar filters to explore different data segments
   - Download reports for stakeholder sharing

## 📈 **Business Impact**

### **Cost Optimization**
- Identify low-profit, high-risk orders for review
- Data-driven carrier selection and negotiation
- Systematic approach to resource allocation

### **Customer Experience**
- Proactive management of high-value, at-risk shipments
- Correlation analysis between delivery and satisfaction
- Automated identification of customers requiring follow-up

### **Strategic Decision Making**
- Unified view for operational prioritization
- Predictive analytics for proactive management
- Executive-ready insights and recommendations

## 🛠️ **Technical Architecture**

### **Data Pipeline**
- **ETL Process**: Automated CSV ingestion and validation
- **Data Integration**: Left joins preserving all order records
- **Metric Calculation**: Profit analysis and business KPIs

### **Machine Learning**
- **Algorithm**: RandomForest with hyperparameter optimization
- **Features**: Priority, category, segment, carrier, distance, traffic, weather
- **Preprocessing**: OneHotEncoder + StandardScaler pipeline
- **Validation**: Cross-validated training on 150+ completed orders

### **Visualization**
- **Framework**: Plotly for interactive charts
- **Dashboard**: Streamlit for web application
- **Filtering**: Real-time data exploration
- **Export**: Professional CSV reports

## 📊 **Platform Screenshots**

### Profit-Risk Quadrant Analysis
The core innovation showing strategic order positioning:
- 🟢 High Profit, Low Risk (Ideal)
- 🟡 High Profit, High Risk (Action Needed)
- 🔵 Low Profit, Low Risk (Optimize)
- 🔴 Low Profit, High Risk (Critical)

### Interactive Dashboard
- Real-time filtering by customer segment, carrier, product category
- Dynamic visualizations updating based on selections
- Executive metrics and KPI tracking

### Download Reports
- Executive summary for leadership presentations
- Detailed analytics for operational teams
- Priority orders for immediate action

## 🏗️ **Project Structure**

```
intelligent-logistics-analytics/
├── nexgen_analytics/
│   ├── app.py                 # Main Streamlit application
│   ├── data_loader.py        # CSV data ingestion and validation
│   ├── data_processor.py     # Data transformation and metrics
│   ├── feature_engineer.py   # ML feature preparation
│   ├── risk_model.py         # RandomForest risk prediction
│   ├── visualizations.py     # Plotly charts and analysis
│   └── requirements.txt      # Python dependencies
├── datasets/                  # CSV data files (not included)
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── DEPLOYMENT_GUIDE.md      # Production deployment guide
```

## 🔧 **Development**

### **Adding New Features**
1. Follow the modular architecture pattern
2. Add new visualizations to `visualizations.py`
3. Extend data processing in `data_processor.py`
4. Update ML features in `feature_engineer.py`

### **Testing**
- Validate data pipeline with sample datasets
- Test ML model performance with cross-validation
- Verify dashboard functionality across browsers

## 📝 **Data Requirements**

The platform expects CSV files with the following structure:

- **orders.csv**: Order details, customer segments, priorities
- **cost_breakdown.csv**: Detailed cost components per order
- **delivery_performance.csv**: Carrier performance and delivery status
- **routes_distance.csv**: Route information and traffic data
- **customer_feedback.csv**: Ratings and satisfaction data
- **vehicle_fleet.csv**: Fleet information and capabilities
- **warehouse_inventory.csv**: Inventory and storage data

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👨‍💻 **Author**

**Davis** - [GitHub Profile](https://github.com/davis18-03)

## 🙏 **Acknowledgments**

- Built with Streamlit for rapid web application development
- Powered by scikit-learn for machine learning capabilities
- Visualizations created with Plotly for interactive charts
- Inspired by real-world logistics optimization challenges

---

⭐ **Star this repository if you found it helpful!**

📧 **Questions or suggestions? Open an issue or reach out!**
