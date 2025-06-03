import streamlit as st
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io

# Set up the page configuration
st.set_page_config(
    page_title="Solar Energy Dashboard",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
    }
    .upload-section {
        background: #e9f5ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌍 Renewable Energy Analytics Dashboard")
st.markdown("""
This interactive dashboard provides comprehensive analysis of solar and wind energy generation, 
consumption patterns, and predictive modeling for optimal energy management.
""")

# Sample data generation function
def generate_sample_data():
    dates = pd.date_range(start="2023-01-01", periods=8281, freq='H')
    consumption = np.sin(np.linspace(0, 100, 8281)) * 150 + 200 + np.random.normal(0, 50, 8281)
    solar = (np.sin((dates.hour-6)/12*np.pi) * 250 * ((dates.hour > 6) & (dates.hour < 18))) + np.random.normal(0, 30, 8281)
    wind = np.random.weibull(2, 8281) * 100 + np.random.normal(0, 20, 8281)
    
    return pd.DataFrame({
        'index': dates,
        'Consumption': np.clip(consumption, 50, 500),
        'Solar': np.clip(solar, 0, 300),
        'Wind': np.clip(wind, 0, 150)
    })

# Data loading and processing
@st.cache_data
def load_data(uploaded_file=None, use_sample=False):
    if use_sample:
        st.warning("Using sample data as no file was uploaded")
        df = generate_sample_data()
        df.set_index('index', inplace=True)
        return df
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        if 'index' not in df.columns:
            st.error("The uploaded file must contain an 'index' column")
            return None
            
        df['index'] = pd.to_datetime(df['index'])
        df.set_index('index', inplace=True)
        
        required_cols = ['Consumption', 'Solar', 'Wind']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
                
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# File upload section
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your energy data (CSV or Excel)", 
        type=['csv', 'xlsx'],
        help="File should contain columns: index, Consumption, Solar, Wind"
    )
    
    use_sample = st.checkbox("Use sample data", value=False)
    
    if uploaded_file is not None or use_sample:
        load_gen_data = load_data(uploaded_file, use_sample)
    else:
        st.warning("Please upload a file or check 'Use sample data'")
        st.stop()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "⏳ Time Analysis", 
    "📈 Distributions", 
    "🤖 Predictive Model", 
    "🔋 Optimization"
])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Data")
        st.dataframe(load_gen_data.head(), height=250)
        st.metric("Total Records", len(load_gen_data))
        
    with col2:
        st.subheader("Key Statistics")
        st.dataframe(load_gen_data.describe(), height=250)

with tab2:
    st.header("Time Series Analysis")
    
    # Resample data
    daily_data = load_gen_data.resample('D').mean()
    weekly_data = load_gen_data.resample('W').mean()
    monthly_data = load_gen_data.resample('M').mean()
    
    # Create plots
    with st.container():
        st.subheader("Daily Patterns")
        fig, ax = plt.subplots(figsize=(12, 4))
        daily_data[['Consumption', 'Solar', 'Wind']].plot(ax=ax)
        ax.set_title("Daily Averages", pad=20)
        ax.grid(True)
        st.pyplot(fig)
    
    with st.container():
        st.subheader("Weekly Patterns")
        fig, ax = plt.subplots(figsize=(12, 4))
        weekly_data[['Consumption', 'Solar', 'Wind']].plot(ax=ax)
        ax.set_title("Weekly Averages", pad=20)
        ax.grid(True)
        st.pyplot(fig)
    
    with st.container():
        st.subheader("Monthly/Seasonal Patterns")
        seasonal_trends = load_gen_data.groupby(load_gen_data.index.month).mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        seasonal_trends[['Consumption', 'Solar', 'Wind']].plot(ax=ax)
        ax.set_title("Monthly Averages", pad=20)
        ax.set_xlabel("Month")
        ax.grid(True)
        st.pyplot(fig)

with tab3:
    st.header("Data Distributions and Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("Consumption Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            load_gen_data['Consumption'].plot(
                kind='hist', 
                bins=30, 
                ax=ax, 
                color='blue', 
                alpha=0.7
            )
            ax.set_xlabel("Consumption (kWh)")
            st.pyplot(fig)
        
    with col2:
        with st.container():
            st.subheader("Solar Generation Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            load_gen_data['Solar'].plot(
                kind='hist', 
                bins=30, 
                ax=ax, 
                color='orange', 
                alpha=0.7
            )
            ax.set_xlabel("Solar Generation (kWh)")
            st.pyplot(fig)
    
    with st.container():
        st.subheader("Wind Generation Distribution")
        fig, ax = plt.subplots(figsize=(12, 4))
        load_gen_data['Wind'].plot(
            kind='hist', 
            bins=30, 
            ax=ax, 
            color='green', 
            alpha=0.7
        )
        ax.set_xlabel("Wind Generation (kWh)")
        st.pyplot(fig)
    
    with st.container():
        st.subheader("Generation vs Consumption Relationship")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(
            load_gen_data['Solar'], 
            load_gen_data['Consumption'], 
            color='orange', 
            alpha=0.5, 
            label='Solar vs Consumption'
        )
        ax.scatter(
            load_gen_data['Wind'], 
            load_gen_data['Consumption'], 
            color='green', 
            alpha=0.5, 
            label='Wind vs Consumption'
        )
        ax.set_xlabel("Generation (kWh)")
        ax.set_ylabel("Consumption (kWh)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

with tab4:
    st.header("Energy Consumption Prediction Model")
    
    # Prepare data
    X = load_gen_data[['Solar', 'Wind']]
    y = load_gen_data['Consumption']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    with col3:
        st.metric("R² Score", f"{r2:.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': ['Solar', 'Wind'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_imp['Feature'], feature_imp['Importance'], color=['orange', 'green'])
    ax.set_title("Feature Importance Scores")
    st.pyplot(fig)
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Consumption (kWh)")
    ax.set_ylabel("Predicted Consumption (kWh)")
    ax.grid(True)
    st.pyplot(fig)

with tab5:
    st.header("Energy Optimization Strategies")
    
    # Calculate hourly averages
    hourly_avg = load_gen_data.resample('H').mean()
    hourly_avg['Net Generation'] = hourly_avg['Solar'] + hourly_avg['Wind'] - hourly_avg['Consumption']
    
    st.subheader("Net Generation Analysis")
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_avg['Net Generation'].plot(ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("Net Generation (Generation - Consumption)")
    ax.set_ylabel("Net Generation (kWh)")
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Energy Storage Simulation")
    
    # Get user input for storage capacity
    storage_capacity = st.slider(
        "Storage Capacity (kWh)", 
        min_value=100, 
        max_value=5000, 
        value=1000, 
        step=100
    )
    
    # Simulate storage
    stored_energy = 0
    energy_flow = []
    storage_levels = []
    
    for net_gen in hourly_avg['Net Generation']:
        if net_gen > 0:
            # Store energy
            energy_to_store = min(storage_capacity - stored_energy, net_gen)
            stored_energy += energy_to_store
            energy_flow.append(energy_to_store)
        else:
            # Use stored energy
            energy_to_use = min(stored_energy, -net_gen)
            stored_energy -= energy_to_use
            energy_flow.append(-energy_to_use)
        storage_levels.append(stored_energy)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly_avg.index, energy_flow, label='Energy Flow')
    ax.plot(hourly_avg.index, storage_levels, label='Storage Level', linestyle='--')
    ax.set_title(f"Energy Storage Simulation (Capacity: {storage_capacity} kWh)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Calculate metrics
    total_surplus = hourly_avg[hourly_avg['Net Generation'] > 0]['Net Generation'].sum()
    total_deficit = -hourly_avg[hourly_avg['Net Generation'] < 0]['Net Generation'].sum()
    utilization = max(storage_levels) / storage_capacity * 100 if storage_capacity > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Energy Surplus", f"{total_surplus:.2f} kWh")
    with col2:
        st.metric("Total Energy Deficit", f"{total_deficit:.2f} kWh")
    with col3:
        st.metric("Storage Utilization", f"{utilization:.2f}%")

# Footer
st.markdown("---")
st.markdown("© 2023 Renewable Energy Analytics Dashboard")