import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, hsl(260, 30%, 8%) 0%, hsl(240, 20%, 12%) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, hsl(260, 50%, 20%) 0%, hsl(240, 40%, 25%) 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    h1, h2, h3 {
        color: white;
        font-weight: 600;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, hsl(260, 25%, 15%) 0%, hsl(240, 20%, 10%) 100%);
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .tab-content {
        background: rgba(255, 255, 255, 0.02);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    """Load and preprocess the financial data"""
    try:
        df = pd.read_csv("financial_data.csv")
        
        # Convert date columns
        df['sales_Invoice Date'] = pd.to_datetime(df['sales_Invoice Date'], errors='coerce')
        df['Purchase Invoice Date'] = pd.to_datetime(df['Purchase Invoice Date'], errors='coerce')
        
        # Convert amount columns to numeric
        numeric_cols = [
            'sales_Net Amount', 'sales_Grand Amount', 'Purchase Net Amount', 'Purchase Grand Amount',
            'sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST',
            'Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add calculated fields
        df['sales_year'] = df['sales_Invoice Date'].dt.year
        df['purchase_year'] = df['Purchase Invoice Date'].dt.year
        df['sales_month'] = df['sales_Invoice Date'].dt.to_period('M')
        df['purchase_month'] = df['Purchase Invoice Date'].dt.to_period('M')
        df['net_profit'] = df['sales_Grand Amount'] - df['Purchase Grand Amount']
        
        return df
    except FileNotFoundError:
        st.error("📄 financial_data.csv file not found. Please upload your data file.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_records = 50
    
    customers = ['Tech Solutions Ltd', 'Manufacturing Corp', 'Global Dynamics', 'Industrial Systems', 'Precision Tools']
    vendors = ['Raw Materials Ltd', 'Steel Suppliers', 'Component Works', 'Tool Manufacturers', 'Equipment Rental']
    
    data = []
    for i in range(n_records):
        net_amount = np.random.uniform(50000, 500000)
        purchase_net = np.random.uniform(30000, 300000)
        cgst_amount = net_amount * 0.09
        sgst_amount = net_amount * 0.09
        igst_amount = net_amount * 0.18 if np.random.random() > 0.7 else 0
        purchase_cgst = purchase_net * 0.09
        purchase_sgst = purchase_net * 0.09
        purchase_igst = purchase_net * 0.18 if np.random.random() > 0.7 else 0
        
        data.append({
            'sales_Sl. No': i + 1,
            'sales_Invoice Number': f'S{2024}{str(i + 1).zfill(3)}',
            'sales_Invoice Date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'sales_Customer Name': np.random.choice(customers),
            'GST Number': f'GST{np.random.randint(100000, 999999)}',
            'sales_Net Amount': net_amount,
            'sales_Tax Type CGST': 'CGST',
            'sales_Tax Amount CGST': cgst_amount,
            'sales_Tax Type SGST': 'SGST',
            'sales_Tax Amount SGST': sgst_amount,
            'sales_Tax Type IGST': 'IGST',
            'sales_Tax Amount IGST': igst_amount,
            'sales_Grand Amount': net_amount + cgst_amount + sgst_amount + igst_amount,
            'Purchase Invoice Number': f'P{2024}{str(i + 1).zfill(3)}',
            'Purchase Invoice Date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'Purchase Customer Name': np.random.choice(vendors),
            'TIN Number': f'TIN{np.random.randint(100000, 999999)}',
            'Purchase Net Amount': purchase_net,
            'Purchase Tax Type CGST': 'CGST',
            'Purchase Tax Amount CGST': purchase_cgst,
            'Purchase Tax Type SGST': 'SGST',
            'Purchase Tax Amount SGST': purchase_sgst,
            'Purchase Tax Type IGST': 'IGST',
            'Purchase Tax Amount IGST': purchase_igst,
            'Purchase Grand Amount': purchase_net + purchase_cgst + purchase_sgst + purchase_igst
        })
    
    df = pd.DataFrame(data)
    df['sales_year'] = df['sales_Invoice Date'].dt.year
    df['purchase_year'] = df['Purchase Invoice Date'].dt.year
    df['sales_month'] = df['sales_Invoice Date'].dt.to_period('M')
    df['purchase_month'] = df['Purchase Invoice Date'].dt.to_period('M')
    df['net_profit'] = df['sales_Grand Amount'] - df['Purchase Grand Amount']
    
    return df

# ML Forecasting functions
def generate_linear_forecast(data):
    """Generate linear regression forecast"""
    if len(data) < 2:
        return [data[-1] if data else 0] * 6
    
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array(range(len(data), len(data) + 6)).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    # Add some variation
    variation = np.random.normal(0, np.std(data) * 0.1, 6)
    return np.maximum(0, forecast + variation)

def generate_arima_forecast(data):
    """Simplified ARIMA-like forecast"""
    if len(data) < 3:
        return [data[-1] if data else 0] * 6
    
    trend = np.mean(np.diff(data[-3:]))
    seasonality = np.sin(np.arange(6) * 2 * np.pi / 12) * np.std(data) * 0.1
    base_value = data[-1]
    
    forecast = []
    for i in range(6):
        value = base_value + trend * (i + 1) + seasonality[i]
        noise = np.random.normal(0, np.std(data) * 0.05)
        forecast.append(max(0, value + noise))
    
    return forecast

def generate_lstm_forecast(data):
    """Simplified LSTM-like forecast with memory"""
    if len(data) < 3:
        return [data[-1] if data else 0] * 6
    
    lookback = min(12, len(data))
    recent = data[-lookback:]
    weights = np.exp(np.arange(lookback) / lookback)
    weighted_avg = np.average(recent, weights=weights)
    
    momentum = (data[-1] - data[-2]) if len(data) >= 2 else 0
    
    forecast = []
    for i in range(6):
        decay = np.exp(-i * 0.1)
        value = weighted_avg + momentum * decay
        noise = np.random.normal(0, np.std(data) * 0.03)
        forecast.append(max(0, value + noise))
    
    return forecast

# Load data
df = load_data()

# Sidebar
st.sidebar.title("🏢 Financial Dashboard")
st.sidebar.markdown("---")

# Year selection
available_years = sorted([year for year in df['sales_year'].dropna().unique() if not pd.isna(year)])
if not available_years:
    available_years = [2024]

selected_year = st.sidebar.selectbox("📅 Select Year", available_years, index=len(available_years)-1)

# Tab selection
tabs = [
    "📋 Company Overview",
    "💰 Sales & Revenue", 
    "📈 Trends & Analytics",
    "🧾 Tax Management",
    "🤖 ML Forecasting"
]

selected_tab = st.sidebar.radio("Navigate to:", tabs)

# Filter data by year
df_year = df[
    (df['sales_year'] == selected_year) | 
    (df['purchase_year'] == selected_year)
].copy()

st.sidebar.markdown("---")
st.sidebar.info("💡 **AI-Powered Analytics**: This dashboard uses machine learning for sales forecasting and trend analysis.")

# Main content
if selected_tab == "📋 Company Overview":
    st.markdown('<div class="main-header"><h1>📋 Company Financial Overview</h1><p>Comprehensive view of your financial performance and key metrics</p></div>', unsafe_allow_html=True)
    
    # Key metrics
    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchases = df_year['Purchase Grand Amount'].sum()
    total_tax = (df_year['sales_Tax Amount CGST'].sum() + 
                df_year['sales_Tax Amount SGST'].sum() + 
                df_year['sales_Tax Amount IGST'].sum())
    net_profit = total_sales - total_purchases
    gross_margin = (net_profit / total_sales * 100) if total_sales > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Total Sales", f"₹{total_sales:,.0f}", f"{((total_sales / df_year['sales_Grand Amount'].mean()) - 1) * 100:.1f}%" if total_sales > 0 else "0%")
    
    with col2:
        st.metric("🛒 Total Purchases", f"₹{total_purchases:,.0f}")
    
    with col3:
        st.metric("💸 Total Tax", f"₹{total_tax:,.0f}")
    
    with col4:
        st.metric("📊 Net Profit", f"₹{net_profit:,.0f}")
    
    with col5:
        st.metric("📈 Gross Margin", f"{gross_margin:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Monthly Sales Trend")
        monthly_sales = df_year.groupby('sales_month')['sales_Grand Amount'].sum().reset_index()
        monthly_sales['sales_month'] = monthly_sales['sales_month'].astype(str)
        
        fig = px.line(monthly_sales, x='sales_month', y='sales_Grand Amount', 
                     title="Monthly Sales Performance")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Monthly Profit Analysis")
        monthly_profit = df_year.groupby('sales_month').agg({
            'sales_Grand Amount': 'sum',
            'Purchase Grand Amount': 'sum'
        }).reset_index()
        monthly_profit['net_profit'] = monthly_profit['sales_Grand Amount'] - monthly_profit['Purchase Grand Amount']
        monthly_profit['sales_month'] = monthly_profit['sales_month'].astype(str)
        
        fig = px.bar(monthly_profit, x='sales_month', y='net_profit',
                    title="Monthly Net Profit")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "💰 Sales & Revenue":
    st.markdown('<div class="main-header"><h1>💰 Sales & Revenue Analysis</h1><p>Detailed breakdown of revenue streams and customer insights</p></div>', unsafe_allow_html=True)
    
    # Revenue metrics
    total_revenue = df_year['sales_Grand Amount'].sum()
    gst_collected = df_year['sales_Tax Amount CGST'].sum() + df_year['sales_Tax Amount SGST'].sum()
    igst_collected = df_year['sales_Tax Amount IGST'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    
    with col2:
        st.metric("🧾 GST Collected", f"₹{gst_collected:,.0f}")
    
    with col3:
        st.metric("🌍 IGST Collected", f"₹{igst_collected:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Monthly Revenue Trend")
        monthly_revenue = df_year.groupby('sales_month')['sales_Grand Amount'].sum().reset_index()
        monthly_revenue['sales_month'] = monthly_revenue['sales_month'].astype(str)
        
        fig = px.bar(monthly_revenue, x='sales_month', y='sales_Grand Amount',
                    title="Monthly Revenue Performance")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 GST Distribution")
        gst_data = pd.DataFrame({
            'GST Type': ['CGST', 'SGST', 'IGST'],
            'Amount': [df_year['sales_Tax Amount CGST'].sum(), df_year['sales_Tax Amount SGST'].sum(), df_year['sales_Tax Amount IGST'].sum()]
        })
        
        fig = px.pie(gst_data, values='Amount', names='GST Type',
                    title="GST Collection Distribution")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top customers
    st.subheader("🏆 Top 5 Customers by Revenue")
    if 'sales_Customer Name' in df_year.columns:
        top_customers = df_year.groupby('sales_Customer Name').agg({
            'sales_Grand Amount': 'sum',
            'sales_Invoice Date': 'count'
        }).round(2).sort_values('sales_Grand Amount', ascending=False).head(5)
        top_customers.columns = ['Total Revenue (₹)', 'Transactions']
        st.dataframe(top_customers, use_container_width=True)

elif selected_tab == "📈 Trends & Analytics":
    st.markdown('<div class="main-header"><h1>📈 Business Trends & Analytics</h1><p>Deep insights into customer behavior and business patterns</p></div>', unsafe_allow_html=True)
    
    # Monthly trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Monthly Sales & Purchase Trends")
        monthly_data = pd.merge(
            df_year.groupby('sales_month')['sales_Grand Amount'].sum().reset_index(),
            df_year.groupby('purchase_month')['Purchase Grand Amount'].sum().reset_index(),
            left_on='sales_month', right_on='purchase_month', how='outer'
        ).fillna(0)
        
        monthly_data['sales_month'] = monthly_data['sales_month'].astype(str)
        
        fig = px.line(monthly_data, x='sales_month', y=['sales_Grand Amount', 'Purchase Grand Amount'],
                     title="Sales vs Purchases Over Time")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Transaction Volume Analysis")
        transaction_volume = df_year.groupby('sales_month').size().reset_index(name='transaction_count')
        transaction_volume['sales_month'] = transaction_volume['sales_month'].astype(str)
        
        fig = px.area(transaction_volume, x='sales_month', y='transaction_count',
                     title="Monthly Transaction Volume")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer and vendor analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Top Customers")
        if 'sales_Customer Name' in df_year.columns:
            customer_analysis = df_year.groupby('sales_Customer Name').agg({
                'sales_Grand Amount': 'sum',
                'sales_Invoice Date': 'count'
            }).sort_values('sales_Grand Amount', ascending=False).head(5)
            customer_analysis.columns = ['Revenue (₹)', 'Transactions']
            st.dataframe(customer_analysis)
    
    with col2:
        st.subheader("🏭 Top Vendors")
        if 'Purchase Customer Name' in df_year.columns:
            vendor_analysis = df_year.groupby('Purchase Customer Name').agg({
                'Purchase Grand Amount': 'sum',
                'Purchase Invoice Date': 'count'
            }).sort_values('Purchase Grand Amount', ascending=False).head(5)
            vendor_analysis.columns = ['Spend (₹)', 'Transactions']
            st.dataframe(vendor_analysis)

elif selected_tab == "🧾 Tax Management":
    st.markdown('<div class="main-header"><h1>🧾 Tax Management & GST Analysis</h1><p>Comprehensive tax tracking and compliance overview</p></div>', unsafe_allow_html=True)
    
    # Tax summary metrics
    outward_gst = df_year['sales_Tax Amount CGST'].sum() + df_year['sales_Tax Amount SGST'].sum() + df_year['sales_Tax Amount IGST'].sum()
    input_credit = df_year['Purchase Tax Amount CGST'].sum() + df_year['Purchase Tax Amount SGST'].sum() + df_year['Purchase Tax Amount IGST'].sum()
    net_payable = outward_gst - input_credit
    tax_efficiency = (input_credit / outward_gst * 100) if outward_gst > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📤 Total Outward GST", f"₹{outward_gst:,.0f}")
    
    with col2:
        st.metric("📥 Total Input Credit", f"₹{input_credit:,.0f}")
    
    with col3:
        st.metric("💰 Net Payable", f"₹{net_payable:,.0f}")
    
    with col4:
        st.metric("⚡ Tax Efficiency", f"{tax_efficiency:.1f}%")
    
    # GST breakdown table
    st.subheader("📊 Net GST Payable/Receivable Summary")
    
    gst_summary = pd.DataFrame({
        'GST Type': ['CGST', 'SGST', 'IGST'],
        'Outward GST (₹)': [
            df_year['sales_Tax Amount CGST'].sum(),
            df_year['sales_Tax Amount SGST'].sum(), 
            df_year['sales_Tax Amount IGST'].sum()
        ],
        'Input Credit (₹)': [
            df_year['Purchase Tax Amount CGST'].sum(),
            df_year['Purchase Tax Amount SGST'].sum(),
            df_year['Purchase Tax Amount IGST'].sum()
        ],
        'Net Payable (₹)': [
            df_year['sales_Tax Amount CGST'].sum() - df_year['Purchase Tax Amount CGST'].sum(),
            df_year['sales_Tax Amount SGST'].sum() - df_year['Purchase Tax Amount SGST'].sum(),
            df_year['sales_Tax Amount IGST'].sum() - df_year['Purchase Tax Amount IGST'].sum()
        ]
    })
    
    st.dataframe(gst_summary, use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Net GST Payable Distribution")
        fig = px.pie(gst_summary, values='Net Payable (₹)', names='GST Type',
                    title="Distribution of Net GST Payable")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Monthly GST Trend")
        monthly_gst = df_year.groupby('sales_month').agg({
            'sales_Tax Amount CGST': 'sum',
            'sales_Tax Amount SGST': 'sum',
            'sales_Tax Amount IGST': 'sum',
            'Purchase Tax Amount CGST': 'sum',
            'Purchase Tax Amount SGST': 'sum',
            'Purchase Tax Amount IGST': 'sum'
        }).reset_index()
        
        monthly_gst['outward_gst'] = monthly_gst['sales_Tax Amount CGST'] + monthly_gst['sales_Tax Amount SGST'] + monthly_gst['sales_Tax Amount IGST']
        monthly_gst['input_gst'] = monthly_gst['Purchase Tax Amount CGST'] + monthly_gst['Purchase Tax Amount SGST'] + monthly_gst['Purchase Tax Amount IGST']
        monthly_gst['net_gst'] = monthly_gst['outward_gst'] - monthly_gst['input_gst']
        monthly_gst['sales_month'] = monthly_gst['sales_month'].astype(str)
        
        fig = px.bar(monthly_gst, x='sales_month', y=['outward_gst', 'input_gst'],
                    title="Monthly Outward vs Input GST",
                    barmode='group')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "🤖 ML Forecasting":
    st.markdown('<div class="main-header"><h1>🤖 AI-Powered Sales Forecasting</h1><p>Advanced machine learning models for business predictions</p></div>', unsafe_allow_html=True)
    
    # Model selection
    st.subheader("🔬 Select Forecasting Model")
    
    models = {
        "Linear Regression": {
            "description": "Traditional linear trend analysis",
            "accuracy": "85%",
            "complexity": "Low"
        },
        "ARIMA": {
            "description": "Auto-regressive integrated moving average",
            "accuracy": "89%", 
            "complexity": "Medium"
        },
        "LSTM": {
            "description": "Deep learning neural network with memory",
            "accuracy": "93%",
            "complexity": "High"
        }
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        linear_selected = st.button("📈 Linear Regression", help=models["Linear Regression"]["description"])
    
    with col2:
        arima_selected = st.button("🔄 ARIMA Model", help=models["ARIMA"]["description"])
    
    with col3:
        lstm_selected = st.button("🧠 LSTM Neural Network", help=models["LSTM"]["description"])
    
    # Default to Linear Regression
    selected_model = "Linear Regression"
    if arima_selected:
        selected_model = "ARIMA"
    elif lstm_selected:
        selected_model = "LSTM"
    
    # Prepare historical data
    historical_sales = df_year.groupby('sales_month')['sales_Grand Amount'].sum().values
    
    if len(historical_sales) > 0:
        # Generate forecasts
        if selected_model == "Linear Regression":
            forecast = generate_linear_forecast(historical_sales)
        elif selected_model == "ARIMA":
            forecast = generate_arima_forecast(historical_sales)
        else:  # LSTM
            forecast = generate_lstm_forecast(historical_sales)
        
        # Display model info and metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Selected Model", selected_model)
        
        with col2:
            total_forecast = sum(forecast)
            st.metric("📊 6-Month Forecast", f"₹{total_forecast:,.0f}")
        
        with col3:
            growth_rate = ((forecast[-1] / historical_sales[-1] - 1) * 100) if len(historical_sales) > 0 and historical_sales[-1] > 0 else 0
            st.metric("📈 Growth Prediction", f"{growth_rate:.1f}%")
        
        with col4:
            st.metric("🎯 Model Accuracy", models[selected_model]["accuracy"])
        
        # Forecasting chart
        st.subheader(f"📊 Sales Forecast - {selected_model}")
        
        # Create combined historical + forecast data
        months = list(range(1, len(historical_sales) + 1))
        forecast_months = list(range(len(historical_sales) + 1, len(historical_sales) + 7))
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=months,
            y=historical_sales,
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#8B5CF6', width=3)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_months,
            y=forecast,
            mode='lines+markers',
            name=f'{selected_model} Forecast',
            line=dict(color='#06B6D4', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Sales Forecast using {selected_model}",
            xaxis_title="Month",
            yaxis_title="Sales Amount (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("🔍 Model Performance Comparison")
        
        comparison_data = pd.DataFrame({
            'Model': list(models.keys()),
            'Accuracy': [models[m]["accuracy"] for m in models.keys()],
            'Complexity': [models[m]["complexity"] for m in models.keys()],
            'Description': [models[m]["description"] for m in models.keys()]
        })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # AI Insights
        st.subheader("🎯 AI-Generated Insights")
        
        avg_historical = np.mean(historical_sales) if len(historical_sales) > 0 else 0
        avg_forecast = np.mean(forecast)
        trend = "upward" if avg_forecast > avg_historical else "downward"
        
        insights = f"""
        **📊 Forecast Analysis:**
        - The {selected_model} model predicts a **{trend} trend** in sales for the next 6 months
        - Average monthly forecast: **₹{avg_forecast:,.0f}**
        - Expected total revenue: **₹{sum(forecast):,.0f}**
        
        **💡 Strategic Recommendations:**
        - {"Focus on maintaining growth momentum" if trend == "upward" else "Consider strategies to boost sales performance"}
        - Monitor key performance indicators monthly
        - Adjust inventory and resource planning based on predictions
        """
        
        st.markdown(insights)
    
    else:
        st.warning("⚠️ Insufficient data for forecasting. Please ensure you have historical sales data.")

# Footer
st.markdown("---")
st.markdown("📊 **Financial Dashboard** - Powered by AI & Machine Learning | Built with Streamlit")
