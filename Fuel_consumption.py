import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Fuel Consumption Forecasting",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3498db;
    }
    .insight-box {
        background-color: #e8f4fc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #2c3e50;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Fuel Consumption Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Introduction
st.write("""
This dashboard helps you forecast fuel consumption for agricultural operations and estimate associated costs.
Adjust the parameters in the sidebar to match your specific operation and see real-time predictions.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Operation Parameters")
    
    area = st.slider("Area Covered (hectares)", 1, 500, 100)
    
    tractor_model = st.selectbox(
        "Tractor Model & HP",
        ["John Deere 6150M (150 HP)", "New Holland T7.270 (270 HP)", 
         "Case IH Steiger 450 (450 HP)", "Massey Ferguson 8700 (250 HP)", 
         "Kubota M7-131 (131 HP)"]
    )
    
    operation_type = st.selectbox(
        "Operation Type",
        ["Plowing", "Harvesting", "Seeding", "Spraying", "Fertilizing"]
    )
    
    soil_condition = st.selectbox(
        "Soil Condition",
        ["Dry", "Normal", "Wet", "Very Wet"]
    )
    
    load_weight = st.slider("Load Weight (kg, if applicable)", 0, 10000, 0, step=500)
    
    fuel_price = st.slider("Fuel Price (per liter $)", 0.5, 3.0, 1.5, step=0.05)
    
    st.info("Adjust the parameters to match your operation and see real-time predictions.")

# Simulation model (in a real application, this would connect to a ML backend)
def predict_fuel_consumption(area, tractor_model, operation_type, soil_condition, load_weight):
    # Base consumption values (liters per hectare)
    base_consumption = {
        "Plowing": 12,
        "Harvesting": 8,
        "Seeding": 5,
        "Spraying": 3,
        "Fertilizing": 4
    }
    
    # Tractor factors (multipliers based on tractor model)
    tractor_factors = {
        "John Deere 6150M (150 HP)": 1.0,
        "New Holland T7.270 (270 HP)": 1.2,
        "Case IH Steiger 450 (450 HP)": 1.5,
        "Massey Ferguson 8700 (250 HP)": 1.1,
        "Kubota M7-131 (131 HP)": 0.9
    }
    
    # Soil condition factors
    soil_factors = {
        "Dry": 0.9,
        "Normal": 1.0,
        "Wet": 1.2,
        "Very Wet": 1.5
    }
    
    # Load factor (if applicable)
    load_factor = 1 + (load_weight / 10000) if load_weight > 0 else 1
    
    # Calculate consumption
    per_hectare = (
        base_consumption[operation_type] * 
        tractor_factors[tractor_model] * 
        soil_factors[soil_condition] * 
        load_factor
    )
    
    total_fuel = per_hectare * area
    
    return per_hectare, total_fuel

# Generate predictions
per_hectare, total_fuel = predict_fuel_consumption(area, tractor_model, operation_type, soil_condition, load_weight)
total_cost = total_fuel * fuel_price

# Display results in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Fuel per Hectare", f"{per_hectare:.2f} L")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Total Fuel", f"{total_fuel:.2f} L")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Estimated Cost", f"${total_cost:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Visualization section
st.markdown('<h2 class="sub-header">Consumption Analysis</h2>', unsafe_allow_html=True)

# Create data for visualizations
op_types = ["Plowing", "Harvesting", "Seeding", "Spraying", "Fertilizing"]
base_values = [12, 8, 5, 3, 4]

# Adjust values based on selected tractor and soil
tractor_factors = {
    "John Deere 6150M (150 HP)": 1.0,
    "New Holland T7.270 (270 HP)": 1.2,
    "Case IH Steiger 450 (450 HP)": 1.5,
    "Massey Ferguson 8700 (250 HP)": 1.1,
    "Kubota M7-131 (131 HP)": 0.9
}

soil_factors = {
    "Dry": 0.9,
    "Normal": 1.0,
    "Wet": 1.2,
    "Very Wet": 1.5
}

adjusted_values = [val * tractor_factors[tractor_model] * soil_factors[soil_condition] for val in base_values]

# Create comparison chart
fig = go.Figure(data=[
    go.Bar(name='Standard Operation', x=op_types, y=base_values, marker_color='lightblue'),
    go.Bar(name='Your Operation', x=op_types, y=adjusted_values, marker_color='royalblue')
])

fig.update_layout(
    title='Fuel Consumption Comparison by Operation Type',
    xaxis_title='Operation Type',
    yaxis_title='Liters per Hectare',
    barmode='group'
)

st.plotly_chart(fig, use_container_width=True)

# Create radar chart for operation analysis
categories = ['Fuel Efficiency', 'Operation Speed', 'Fuel Cost', 'Productivity', 'Environmental Impact']

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[70, 80, 65, 85, 60],
    theta=categories,
    fill='toself',
    name='Optimal Operation'
))

fig_radar.add_trace(go.Scatterpolar(
    r=[60, 70, 75, 75, 50],
    theta=categories,
    fill='toself',
    name='Your Operation'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )),
    title='Operation Efficiency Analysis',
    showlegend=True
)

st.plotly_chart(fig_radar, use_container_width=True)

# Insights and recommendations
st.markdown('<h2 class="sub-header">Insights & Recommendations</h2>', unsafe_allow_html=True)

if per_hectare > 15:
    st.warning("""
    **High Fuel Consumption Detected**
    
    Your operation is using more fuel than typical for this operation type. Consider:
    - Checking equipment maintenance
    - Optimizing operation speed
    - Evaluating different equipment options
    """)
elif per_hectare < 5:
    st.success("""
    **Excellent Fuel Efficiency**
    
    Your operation is using less fuel than typical for this operation type. Good job!
    """)
else:
    st.info("""
    **Normal Fuel Consumption**
    
    Your fuel consumption is within expected range for this operation type.
    """)

if soil_condition == "Very Wet":
    st.warning("""
    **Soil Condition Impact**
    
    Very wet soil conditions can increase fuel consumption by up to 50%. 
    Consider waiting for better conditions if possible.
    """)

if "450" in tractor_model:
    st.info("""
    **High-Power Equipment**
    
    Your high-power tractor is excellent for large operations but consumes more fuel. 
    Ensure you're using it for appropriate tasks.
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>Fuel Consumption Forecasting Dashboard &copy; 2023</p>
    <p><em>Note: This dashboard uses a simulation for forecasting. In a real application, this would connect to a backend with machine learning models for more accurate predictions.</em></p>
</div>
""", unsafe_allow_html=True)




