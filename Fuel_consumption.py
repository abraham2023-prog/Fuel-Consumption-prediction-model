import streamlit as st
import pandas as pd
import numpy as np
import random

# Set page configuration
st.set_page_config(
    page_title="Synthetic Fuel Data Generator",
    page_icon="⛽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .highlight {
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
st.markdown('<h1 class="main-header">Synthetic Fuel Consumption Dataset Generator</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="highlight">
This tool generates a synthetic dataset for fuel consumption forecasting in agricultural operations.
You can customize the dataset size and parameters to match your needs.
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("Dataset Configuration")
    
    num_records = st.slider("Number of Records", 100, 10000, 1000)
    
    st.subheader("Tractor Models")
    tractor_options = {
        "John Deere 6150M (150 HP)": 1.0,
        "New Holland T7.270 (270 HP)": 1.2,
        "Case IH Steiger 450 (450 HP)": 1.5,
        "Massey Ferguson 8700 (250 HP)": 1.1,
        "Kubota M7-131 (131 HP)": 0.9
    }
    selected_tractors = {}
    for tractor, factor in tractor_options.items():
        selected_tractors[tractor] = st.checkbox(tractor, value=True)
    
    st.subheader("Operation Types")
    operation_options = {
        "Plowing": 12,
        "Harvesting": 8,
        "Seeding": 5,
        "Spraying": 3,
        "Fertilizing": 4
    }
    selected_operations = {}
    for operation, base_cons in operation_options.items():
        selected_operations[operation] = st.checkbox(operation, value=True)
    
    st.subheader("Soil Conditions")
    soil_options = {
        "Dry": 0.9,
        "Normal": 1.0,
        "Wet": 1.2,
        "Very Wet": 1.5
    }
    selected_soils = {}
    for soil, factor in soil_options.items():
        selected_soils[soil] = st.checkbox(soil, value=True)
    
    add_noise = st.checkbox("Add realistic noise/variation", value=True)
    include_load = st.checkbox("Include load weight data", value=True)

# Generate the dataset
def generate_synthetic_data(num_records, tractors, operations, soils, add_noise, include_load):
    records = []
    
    # Filter selected options
    selected_tractor_list = [t for t, selected in tractors.items() if selected]
    selected_operation_list = [o for o, selected in operations.items() if selected]
    selected_soil_list = [s for s, selected in soils.items() if selected]
    
    if not selected_tractor_list or not selected_operation_list or not selected_soil_list:
        st.error("Please select at least one option in each category")
        return None
    
    for _ in range(num_records):
        # Randomly select parameters
        tractor = random.choice(selected_tractor_list)
        operation = random.choice(selected_operation_list)
        soil = random.choice(selected_soil_list)
        
        # Base consumption for the operation type
        base_consumption = operation_options[operation]
        
        # Apply tractor factor
        tractor_factor = tractor_options[tractor]
        
        # Apply soil factor
        soil_factor = soil_options[soil]
        
        # Generate area (hectares)
        area = round(random.uniform(10, 500), 2)
        
        # Generate load weight if included
        load_weight = 0
        load_factor = 1.0
        if include_load and random.random() > 0.3:  # 70% of records have load
            load_weight = random.randint(500, 5000)
            load_factor = 1 + (load_weight / 10000)
        
        # Calculate base fuel consumption
        fuel_per_hectare = base_consumption * tractor_factor * soil_factor * load_factor
        
        # Add noise if requested
        if add_noise:
            noise = random.normalvariate(1, 0.1)  # 10% noise
            fuel_per_hectare *= noise
        
        # Ensure reasonable values
        fuel_per_hectare = max(1, round(fuel_per_hectare, 2))
        
        # Calculate total fuel
        total_fuel = round(fuel_per_hectare * area, 2)
        
        # Create record
        record = {
            "area_hectares": area,
            "tractor_model": tractor,
            "operation_type": operation,
            "soil_condition": soil,
            "load_weight_kg": load_weight,
            "fuel_per_hectare": fuel_per_hectare,
            "total_fuel_liters": total_fuel
        }
        
        records.append(record)
    
    return pd.DataFrame(records)

# Generate and display dataset
if st.button("Generate Dataset"):
    with st.spinner("Generating synthetic data..."):
        df = generate_synthetic_data(
            num_records, 
            selected_tractors, 
            selected_operations, 
            selected_soils, 
            add_noise, 
            include_load
        )
    
    if df is not None:
        st.success(f"Successfully generated {len(df)} records!")
        
        # Show dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Records", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            avg_fuel = df['fuel_per_hectare'].mean()
            st.metric("Avg Fuel per Hectare", f"{avg_fuel:.2f} L")
        
        # Show first few rows
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Show statistics
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe())
        
        # Show distribution of categorical variables
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tractor_counts = df['tractor_model'].value_counts()
            fig = px.bar(
                x=tractor_counts.index, 
                y=tractor_counts.values,
                title="Tractor Model Distribution",
                labels={'x': 'Tractor Model', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            operation_counts = df['operation_type'].value_counts()
            fig = px.pie(
                values=operation_counts.values, 
                names=operation_counts.index,
                title="Operation Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            soil_counts = df['soil_condition'].value_counts()
            fig = px.bar(
                x=soil_counts.index, 
                y=soil_counts.values,
                title="Soil Condition Distribution",
                labels={'x': 'Soil Condition', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col4:
            fig = px.histogram(
                df, 
                x='fuel_per_hectare',
                title="Fuel per Hectare Distribution",
                labels={'fuel_per_hectare': 'Liters per Hectare'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.subheader("Download Dataset")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="fuel_consumption_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
<div class="footer">
    <p>Synthetic Fuel Consumption Dataset Generator &copy; 2023</p>
    <p><em>This synthetic data can be used to test and demonstrate the Fuel Consumption Forecasting Dashboard.</em></p>
</div>
""", unsafe_allow_html=True)




