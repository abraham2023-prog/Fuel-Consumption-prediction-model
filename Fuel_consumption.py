import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import io

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
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #3498db;
    }
    .prediction-card {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Fuel Consumption Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Introduction
st.write("""
This dashboard helps you forecast fuel consumption for agricultural operations and estimate associated costs.
You can upload historical data, train machine learning models, and make predictions based on your specific operation parameters.
""")

# Initialize session state for ML models and data
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'input_params' not in st.session_state:
    st.session_state.input_params = []

# Generate synthetic data if no data is uploaded
def generate_synthetic_data():
    np.random.seed(42)
    num_records = 500
    
    tractor_models = [
        "John Deere 6150M (150 HP)", 
        "New Holland T7.270 (270 HP)", 
        "Case IH Steiger 450 (450 HP)", 
        "Massey Ferguson 8700 (250 HP)", 
        "Kubota M7-131 (131 HP)"
    ]
    
    operation_types = ["Plowing", "Harvesting", "Seeding", "Spraying", "Fertilizing"]
    soil_conditions = ["Dry", "Normal", "Wet", "Very Wet"]
    
    base_consumption = {
        "Plowing": 12,
        "Harvesting": 8,
        "Seeding": 5,
        "Spraying": 3,
        "Fertilizing": 4
    }
    
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
    
    records = []
    
    for _ in range(num_records):
        tractor = np.random.choice(tractor_models)
        operation = np.random.choice(operation_types)
        soil = np.random.choice(soil_conditions)
        
        area = round(np.random.uniform(10, 500), 2)
        
        if operation in ["Harvesting", "Fertilizing"] and np.random.random() > 0.3:
            load = np.random.randint(500, 5000)
        else:
            load = 0
            
        base_cons = base_consumption[operation]
        tractor_factor = tractor_factors[tractor]
        soil_factor = soil_factors[soil]
        load_factor = 1 + (load / 10000)
        
        noise = np.random.normal(1, 0.1)
        fph = base_cons * tractor_factor * soil_factor * load_factor * noise
        fph = max(1, round(fph, 2))
        
        total = round(fph * area, 2)
        
        records.append({
            "area_hectares": area,
            "tractor_model": tractor,
            "operation_type": operation,
            "soil_condition": soil,
            "load_weight_kg": load,
            "fuel_per_hectare": fph,
            "total_fuel_liters": total
        })
    
    return pd.DataFrame(records)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
                               ["Data Upload", "Model Training", "Prediction", "Model Comparison", "Prediction History"])

# Data Upload Section
if app_mode == "Data Upload":
    st.markdown('<h2 class="sub-header">Upload Historical Data</h2>', unsafe_allow_html=True)
    
    st.info("""
    Upload a CSV file with historical fuel consumption data. The file should include columns for:
    - Area covered (hectares)
    - Tractor model
    - Operation type
    - Soil condition
    - Load weight (if available)
    - Fuel consumption (liters per hectare or total liters)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.historical_data = df
            
            # Display basic information about the dataset
            st.success("Data successfully uploaded!")
            st.write("**Dataset Overview:**")
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1]}")
            
            # Show first few rows
            st.write("**First 5 rows of the dataset:**")
            st.dataframe(df.head())
            
            # Show column information
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.values,
                'Missing Values': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
            
            # Show some basic visualizations of the data
            st.write("**Data Distribution**")
            
            # Numeric columns distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    selected_num_col = st.selectbox("Select numeric column to visualize", numeric_cols)
                    if selected_num_col:
                        fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show correlation if enough numeric columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, title="Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Categorical columns distribution
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                selected_cat_col = st.selectbox("Select categorical column to visualize", categorical_cols)
                if selected_cat_col:
                    value_counts = df[selected_cat_col].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Distribution of {selected_cat_col}",
                                labels={'x': selected_cat_col, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("No data uploaded. Using synthetic data for demonstration.")
        if st.button("Generate Synthetic Data"):
            df = generate_synthetic_data()
            st.session_state.historical_data = df
            
            st.success("Synthetic data generated!")
            st.write("**First 5 rows of the synthetic dataset:**")
            st.dataframe(df.head())
            
            # Show distribution of synthetic data
            st.write("**Synthetic Data Distribution**")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='fuel_per_hectare', title="Distribution of Fuel per Hectare")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                tractor_counts = df['tractor_model'].value_counts()
                fig = px.bar(x=tractor_counts.index, y=tractor_counts.values, 
                            title="Tractor Model Distribution",
                            labels={'x': 'Tractor Model', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

# Model Training Section
elif app_mode == "Model Training":
    st.markdown('<h2 class="sub-header">Train Machine Learning Models</h2>', unsafe_allow_html=True)
    
    if st.session_state.historical_data is None:
        st.warning("No data available. Please upload data or generate synthetic data first.")
    else:
        df = st.session_state.historical_data
        
        # Select target variable
        st.write("### Select Target Variable")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric columns found in the dataset. Please upload a dataset with numeric columns.")
        else:
            target_variable = st.selectbox("Choose the target variable (fuel consumption)", numeric_columns)
            
            # Select features
            st.write("### Select Features")
            feature_columns = st.multiselect("Choose features for the model", 
                                            [col for col in df.columns if col != target_variable],
                                            default=[col for col in df.columns if col != target_variable and df[col].nunique() < 20])
            
            if not feature_columns:
                st.warning("Please select at least one feature.")
            else:
                # Prepare the data
                X = df[feature_columns].copy()
                y = df[target_variable]
                
                # Handle categorical variables
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                for col in categorical_cols:
                    le = LabelEncoder()
                    # Handle missing values and convert to string
                    X[col] = X[col].fillna('Unknown').astype(str)
                    X[col] = le.fit_transform(X[col])
                    st.session_state.encoders[col] = le
                
                # Split the data
                test_size = st.slider("Test Set Size (%)", 10, 40, 20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                
                # Select models to train
                st.write("### Select Models to Train")
                col1, col2 = st.columns(2)
                
                with col1:
                    train_lr = st.checkbox("Linear Regression", value=True)
                
                with col2:
                    train_rf = st.checkbox("Random Forest", value=True)
                
                # Train models
                if st.button("Train Models"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    models = {}
                    results = {}
                    predictions = {}
                    
                    # Linear Regression
                    if train_lr:
                        status_text.text("Training Linear Regression...")
                        lr = LinearRegression()
                        lr.fit(X_train, y_train)
                        models['Linear Regression'] = lr
                        
                        # Evaluate
                        y_pred = lr.predict(X_test)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        results['Linear Regression'] = {'MAE': mae, 'R2': r2, 'RMSE': rmse}
                        predictions['Linear Regression'] = y_pred
                        progress_bar.progress(50)
                    
                    # Random Forest
                    if train_rf:
                        status_text.text("Training Random Forest...")
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        models['Random Forest'] = rf
                        
                        # Evaluate
                        y_pred = rf.predict(X_test)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        results['Random Forest'] = {'MAE': mae, 'R2': r2, 'RMSE': rmse}
                        predictions['Random Forest'] = y_pred
                        progress_bar.progress(100)
                    
                    # Store models and results
                    st.session_state.trained_models = models
                    st.session_state.model_results = results
                    st.session_state.test_predictions = predictions
                    st.session_state.y_test = y_test
                    st.session_state.feature_columns = feature_columns
                    st.session_state.target_variable = target_variable
                    
                    status_text.text("Training completed!")
                    
                    # Display results
                    st.write("### Model Performance")
                    results_df = pd.DataFrame.from_dict(results, orient='index')
                    st.dataframe(results_df.style.format("{:.3f}"))
                    
                    # Visualize model performance
                    st.write("### Prediction vs Actual Values")
                    
                    # Create tabs for different models
                    model_tabs = st.tabs(list(models.keys()))
                    
                    for i, (model_name, model) in enumerate(models.items()):
                        with model_tabs[i]:
                            # Create scatter plot of predicted vs actual values
                            y_pred = predictions[model_name]
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=y_test,
                                y=y_pred,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='blue', size=8, opacity=0.6)
                            ))
                            
                            # Add perfect prediction line
                            max_val = max(max(y_test), max(y_pred))
                            min_val = min(min(y_test), min(y_pred))
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'{model_name} - Predicted vs Actual Values',
                                xaxis_title='Actual Values',
                                yaxis_title='Predicted Values',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Residual plot
                            residuals = y_test - y_pred
                            fig2 = go.Figure()
                            
                            fig2.add_trace(go.Scatter(
                                x=y_pred,
                                y=residuals,
                                mode='markers',
                                name='Residuals',
                                marker=dict(color='green', size=8, opacity=0.6)
                            ))
                            
                            # Add zero residual line
                            fig2.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[0, 0],
                                mode='lines',
                                name='Zero Residual',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig2.update_layout(
                                title=f'{model_name} - Residual Plot',
                                xaxis_title='Predicted Values',
                                yaxis_title='Residuals (Actual - Predicted)',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    # Feature importance for Random Forest
                    if 'Random Forest' in models:
                        st.write("### Feature Importance (Random Forest)")
                        feature_importance = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': models['Random Forest'].feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Create bar chart with proper data
                        fig = px.bar(
                            feature_importance, 
                            x='importance', 
                            y='feature', 
                            orientation='h',
                            title='Feature Importance for Random Forest Model'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download trained models
                    st.write("### Download Trained Models")
                    for model_name, model in models.items():
                        buffer = io.BytesIO()
                        joblib.dump(model, buffer)
                        st.download_button(
                            label=f"Download {model_name}",
                            data=buffer.getvalue(),
                            file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
                            mime="application/octet-stream"
                        )

# Prediction Section
elif app_mode == "Prediction":
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("Please train models first in the 'Model Training' section.")
    else:
        # Get the best model based on R2 score
        best_model_name = max(st.session_state.model_results.items(), key=lambda x: x[1]['R2'])[0]
        best_model = st.session_state.trained_models[best_model_name]
        
        st.info(f"Using {best_model_name} for predictions (best performing model)")
        
        # Create input form based on feature columns
        st.write("### Input Parameters")
        input_data = {}
        
        # Create columns for better layout
        cols = st.columns(2)
        col_idx = 0
        
        for feature in st.session_state.feature_columns:
            with cols[col_idx]:
                if feature in st.session_state.encoders:
                    # Categorical feature - get original values
                    le = st.session_state.encoders[feature]
                    options = le.classes_
                    input_data[feature] = st.selectbox(feature, options)
                else:
                    # Numerical feature
                    # Try to get min and max from historical data if available
                    df = st.session_state.historical_data
                    if df is not None and feature in df.columns:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        default_val = float(df[feature].mean())
                        input_data[feature] = st.slider(feature, min_val, max_val, default_val)
                    else:
                        input_data[feature] = st.number_input(feature, value=0.0)
            
            col_idx = (col_idx + 1) % 2
        
        # Add fuel price for cost calculation
        fuel_price = st.slider("Fuel Price (per liter $)", 0.5, 3.0, 1.5, step=0.05)
        
        # Make prediction
        if st.button("Predict Fuel Consumption"):
            # Prepare input data for prediction
            prediction_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in prediction_df.columns:
                if col in st.session_state.encoders:
                    le = st.session_state.encoders[col]
                    # Handle unseen labels
                    prediction_df[col] = prediction_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    prediction_df[col] = le.transform(prediction_df[col])
            
            # Make prediction
            prediction = best_model.predict(prediction_df)[0]
            
            # Store prediction and input parameters
            st.session_state.predictions.append(prediction)
            st.session_state.input_params.append(input_data.copy())
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.metric("Predicted Consumption", f"{prediction:.2f} {st.session_state.target_variable}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # If we're predicting per hectare and have area, calculate total
            if 'area' in input_data and 'hectare' in st.session_state.target_variable.lower():
                total_fuel = prediction * input_data['area']
                with col2:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.metric("Total Fuel", f"{total_fuel:.2f} L")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                total_cost = total_fuel * fuel_price
                with col3:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.metric("Estimated Cost", f"${total_cost:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate insights
            st.write("### Insights")
            
            # Compare with average from historical data
            if st.session_state.historical_data is not None:
                avg_consumption = st.session_state.historical_data[st.session_state.target_variable].mean()
                
                if prediction > avg_consumption * 1.2:
                    st.warning(f"""
                    **Higher than Average Consumption**
                    
                    Your predicted consumption is {((prediction/avg_consumption)-1)*100:.1f}% higher than the historical average.
                    Consider optimizing your operation parameters.
                    """)
                elif prediction < avg_consumption * 0.8:
                    st.success(f"""
                    **Lower than Average Consumption**
                    
                    Your predicted consumption is {((1-prediction/avg_consumption))*100:.1f}% lower than the historical average.
                    Good job on selecting efficient parameters!
                    """)
                else:
                    st.info(f"""
                    **Average Consumption**
                    
                    Your predicted consumption is close to the historical average.
                    """)
            
            # Show prediction distribution compared to historical data
            st.write("### Prediction Distribution")
            
            fig = go.Figure()
            
            # Add historical data distribution
            fig.add_trace(go.Histogram(
                x=st.session_state.historical_data[st.session_state.target_variable],
                name='Historical Data',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Add current prediction as a vertical line
            fig.add_trace(go.Scatter(
                x=[prediction, prediction],
                y=[0, 100],
                mode='lines',
                name='Your Prediction',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title='Your Prediction vs Historical Distribution',
                xaxis_title=st.session_state.target_variable,
                yaxis_title='Frequency',
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Model Comparison Section
elif app_mode == "Model Comparison":
    st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("Please train models first in the 'Model Training' section.")
    else:
        # Display model performance comparison
        st.write("### Model Performance Metrics")
        results_df = pd.DataFrame.from_dict(st.session_state.model_results, orient='index')
        st.dataframe(results_df.style.format("{:.3f}"))
        
        # Visual comparison
        fig = go.Figure()
        
        for model_name, metrics in st.session_state.model_results.items():
            fig.add_trace(go.Bar(
                name=model_name,
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f'{v:.3f}' for v in metrics.values()],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis_title='Metric Value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance comparison
        st.write("### Feature Importance Comparison")
        
        # Check if we have tree-based models
        tree_models = {name: model for name, model in st.session_state.trained_models.items() 
                      if hasattr(model, 'feature_importances_')}
        
        if tree_models:
            fig = make_subplots(rows=1, cols=len(tree_models), 
                               subplot_titles=list(tree_models.keys()))
            
            for i, (model_name, model) in enumerate(tree_models.items(), 1):
                feature_importance = pd.DataFrame({
                    'feature': st.session_state.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig.add_trace(
                    go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        name=model_name
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tree-based models available for feature importance comparison.")

# Prediction History Section
elif app_mode == "Prediction History":
    st.markdown('<h2 class="sub-header">Prediction History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.predictions:
        st.warning("No predictions made yet. Please make some predictions in the 'Prediction' section.")
    else:
        # Display prediction history
        st.write("### Your Prediction History")
        
        # Create a DataFrame of predictions
        history_df = pd.DataFrame({
            'Prediction': st.session_state.predictions,
            'Target': st.session_state.target_variable
        })
        
        # Add input parameters
        for i, params in enumerate(st.session_state.input_params):
            for key, value in params.items():
                history_df.loc[i, key] = value
        
        st.dataframe(history_df)
        
        # Visualize prediction history
        st.write("### Prediction Trend")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(st.session_state.predictions) + 1)),
            y=st.session_state.predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add average line if we have historical data
        if st.session_state.historical_data is not None:
            avg_consumption = st.session_state.historical_data[st.session_state.target_variable].mean()
            fig.add_trace(go.Scatter(
                x=[1, len(st.session_state.predictions)],
                y=[avg_consumption, avg_consumption],
                mode='lines',
                name='Historical Average',
                line=dict(color='red', dash='dash', width=2)
            ))
        
        fig.update_layout(
            title='Your Prediction History',
            xaxis_title='Prediction Number',
            yaxis_title=st.session_state.target_variable,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution of predictions
        st.write("### Prediction Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=st.session_state.predictions,
            name='Your Predictions',
            nbinsx=15,
            opacity=0.7
        ))
        
        # Add historical data distribution if available
        if st.session_state.historical_data is not None:
            fig.add_trace(go.Histogram(
                x=st.session_state.historical_data[st.session_state.target_variable],
                name='Historical Data',
                nbinsx=30,
                opacity=0.5
            ))
        
        fig.update_layout(
            title='Distribution of Your Predictions',
            xaxis_title=st.session_state.target_variable,
            yaxis_title='Frequency',
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download prediction history
        st.write("### Download Prediction History")
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
<div class="footer">
    <p>Fuel Consumption Forecasting Dashboard with ML Integration (AB) &copy; 2025</p>
    <p><em>This dashboard includes machine learning capabilities for accurate fuel consumption predictions.</em></p>
</div>
""", unsafe_allow_html=True)




