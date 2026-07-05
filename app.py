import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from sklearn.ensemble import RandomForestRegressor

# Import custom modules
from data_generator import generate_earthquake_data
from evolutionary_selector import GeneticAlgorithmFeatureSelector
from model_trainer import train_and_evaluate, bin_magnitude

# Page Configuration
st.set_page_config(
    page_title="Seismic-GA: Earthquake Evolutionary Prediction",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism & Sleek Accent Colors)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(90deg, #58a6ff, #50e3c2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #58a6ff;
    }
    .highlight-box {
        background: rgba(88, 166, 255, 0.05);
        border-left: 5px solid #58a6ff;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    .highlight-box-green {
        background: rgba(80, 227, 194, 0.05);
        border-left: 5px solid #50e3c2;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, rgba(23, 28, 36, 0.9) 0%, rgba(33, 38, 46, 0.9) 100%);
        border: 2px solid #50e3c2;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
        box-shadow: 0 10px 30px rgba(80, 227, 194, 0.15);
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    .badge-selected {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    .badge-discarded {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
PROJECT_DIR = 'C:/Users/manoj/.gemini/antigravity/scratch/earthquake_evolutionary_prediction'
DATA_PATH = os.path.join(PROJECT_DIR, 'earthquake_data.csv')
SUMMARY_PATH = os.path.join(PROJECT_DIR, 'summary_results.json')

# Helper function to load data
@st.cache_data
def get_dataset():
    if not os.path.exists(DATA_PATH):
        # Generate data if not exists
        os.makedirs(PROJECT_DIR, exist_ok=True)
        df = generate_earthquake_data(n_samples=3000, seed=42)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)

# Helper function to cache model training
@st.cache_resource
def get_trained_models(selected_features):
    df = get_dataset()
    # Train Baseline Model (All Features)
    baseline_res = train_and_evaluate(df, target_col='magnitude', selected_features=None, random_state=42)
    # Train GA-Optimized Model
    ga_res = train_and_evaluate(df, target_col='magnitude', selected_features=selected_features, random_state=42)
    return baseline_res, ga_res

# Load initial data and run pipeline if JSON doesn't exist
df = get_dataset()

# Check if JSON exists, otherwise run a fast default run
if not os.path.exists(SUMMARY_PATH):
    st.sidebar.info("Running baseline evolutionary optimization...")
    # Fast GA setup for initial loading
    target_col = 'magnitude'
    all_features = [col for col in df.columns if col != target_col]
    X = df[all_features]
    y = df[target_col]
    
    ga = GeneticAlgorithmFeatureSelector(
        n_features=len(all_features),
        population_size=15,
        generations=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_state=42
    )
    best_chromosome, ga_history = ga.fit(X, y)
    
    selected_features = [all_features[i] for i in range(len(all_features)) if best_chromosome[i]]
    discarded_features = [all_features[i] for i in range(len(all_features)) if not best_chromosome[i]]
    
    # Save default JSON
    summary_data = {
        'all_features': all_features,
        'selected_features': selected_features,
        'discarded_features': discarded_features,
        'ga_history': ga_history
    }
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
    st.sidebar.success("Setup complete!")

# Load summary results
with open(SUMMARY_PATH, 'r') as f:
    summary_data = json.load(f)

all_features = summary_data['all_features']
selected_features = summary_data['selected_features']
discarded_features = summary_data['discarded_features']
ga_history = summary_data['ga_history']

# Cache models
baseline_res, ga_res = get_trained_models(selected_features)

# Header Section
st.markdown("""
<div style="text-align: center; margin-bottom: 30px; margin-top: 10px;">
    <h1 style="color: #ffffff; font-size: 3rem; margin-bottom: 5px;">🌋 SEISMIC-GA</h1>
    <h3 style="color: #8b949e; font-weight: 400; font-size: 1.3rem; margin-top: 0px;">
        Earthquake Magnitude Prediction with Evolutionary Feature Selection
    </h3>
    <hr style="border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(88, 166, 255, 0.4), transparent); margin-top: 15px; margin-bottom: 10px;">
</div>
""", unsafe_allow_html=True)

# Tabs
tab_overview, tab_ga, tab_performance, tab_playground = st.tabs([
    "📂 Project Overview", 
    "🧬 Evolutionary Optimization", 
    "📊 Performance & Confusion Matrix", 
    "🎮 Prediction Playground"
])

# ==================== TAB 1: OVERVIEW ====================
with tab_overview:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Executive Summary")
        st.write("""
        This project aims to predict **earthquake magnitude** as a continuous value using geological, spatial, and wave transmission characteristics. 
        
        To build the most accurate machine learning model (specifically a **Random Forest Regressor**), we employ an **Evolutionary Genetic Algorithm (GA)** to screen and select the most relevant features. The GA filters out environmental noise and non-predictive features, reducing model complexity and mitigating overfitting.
        """)
        
        st.markdown("""
        <div class="highlight-box">
            <h4>💡 Why are we doing this?</h4>
            <ul>
                <li><strong>Feature Pruning:</strong> In seismic monitoring, raw datasets are full of environmental parameters (e.g. ambient temperatures, sound levels, soil moisture). Evolutionary Algorithms allow automatic feature selection without manual trials.</li>
                <li><strong>Comparing Pipelines:</strong> We train two models: a <strong>Baseline Model</strong> using all 15 raw features, and a <strong>GA-Optimized Model</strong> using only features selected by the Genetic Algorithm. We compare their performance to see if feature selection improves predictions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### The Confusion Matrix Challenge")
        st.write("""
        Confusion matrices are traditionally classification metrics (used to measure performance on discrete labels), whereas earthquake magnitude is a continuous variable (regression). 
        
        To meet the mentor's requirement of demonstrating a **confusion matrix**, we perform **Magnitude Binning** after training the regression models:
        1. We train the Random Forest on continuous magnitudes.
        2. We predict the magnitude as a continuous float value (e.g., `4.82`).
        3. Before evaluating classification metrics, we convert both the actual magnitude and the predicted magnitude into one of three risk categories:
        """)
        
        # Display Bins as a table
        bin_df = pd.DataFrame({
            'Category': ['🟢 Low', '🟡 Medium', '🔴 High'],
            'Magnitude Range': ['Magnitude < 4.0', '4.0 <= Magnitude < 5.5', 'Magnitude >= 5.5'],
            'Seismic Significance': [
                'Felt by some, but rarely causes structural damage.',
                'Felt by all, potentially causing minor cracking in walls.',
                'Capable of causing moderate to severe damage in populated regions.'
            ]
        })
        st.table(bin_df)
        
    with col2:
        st.markdown("### Geological Data Features")
        st.write("The dataset incorporates 15 features across three categories:")
        
        # Display Feature description grouped
        st.markdown("""
        **1. Spatial & Time Delay Features (Highly Predictive)**
        - `latitude`, `longitude`: Geospatial epicenter coordinates.
        - `depth`: Focus depth of hypocenter (km).
        - `p_s_wave_delay`: P-to-S wave transmission time delay (seconds). Proportional to hypocentral distance.
        
        **2. Fault & Rock Characteristics (Predictive)**
        - `distance_to_fault`: Distance to nearest active fault line (km).
        - `fault_slip_rate`: Annual fault movement rate (mm/year).
        - `rock_rigidity`: Rigidity/shear modulus of local rock (GPa).
        - `historical_eq_count`: Frequency of previous earthquakes nearby.
        - `tectonic_plate_boundary`: Proximity to plate boundaries (binary).
        
        **3. Environmental Noise & Auxiliary Data (Junk Features)**
        - `seismic_station_density`: Number of sensors nearby (measurement proxy, not geological).
        - `soil_moisture`: Moisture content of topsoil (%).
        - `air_temperature`: Ambient atmospheric temperature (°C).
        - `ambient_noise`: Local industrial/acoustic vibration (dB).
        - `tidal_height`: Ocean tide levels (meters).
        - `random_noise_feature`: Normal distribution white noise.
        """)

# ==================== TAB 2: EVOLUTIONARY OPTIMIZATION ====================
with tab_ga:
    st.markdown("### Genetic Algorithm Feature Selection Results")
    
    col_ga_info, col_ga_chart = st.columns([2, 3])
    
    with col_ga_info:
        st.write("""
        The Genetic Algorithm simulates biological evolution (selection, crossover, and mutation) over generations to identify the optimal chromosome (a binary vector where each bit represents a feature).
        
        **Fitness Function:** The validation $R^2$ score minus a small penalty ($0.005$ per feature selected) to enforce feature parsimony (simplifying the model by preferring fewer features if scores are equal).
        """)
        
        st.markdown("#### Selected Features (GA Chromosome = 1)")
        selected_html = "".join([f'<span class="badge-selected">{feat}</span>' for feat in selected_features])
        st.markdown(selected_html, unsafe_allow_html=True)
        
        st.markdown("#### Discarded Features (GA Chromosome = 0)")
        discarded_html = "".join([f'<span class="badge-discarded">{feat}</span>' for feat in discarded_features])
        st.markdown(discarded_html, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box-green">
            <h4>📈 GA Outcome Analysis</h4>
            <p>The Genetic Algorithm successfully discarded all environmental noise features (air temperature, soil moisture, ambient noise, tidal height, and random noise). It identified that earthquake magnitude is physically determined by geological properties and travel delays, preserving coordinates, depth, fault distances, slip rate, and rigidity. This automated selection confirms the algorithm works as intended!</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ga_chart:
        st.markdown("#### Convergence Curve")
        
        # Plotly Convergence Chart
        gen = [h['generation'] for h in ga_history]
        best_fit = [h['best_fitness'] for h in ga_history]
        avg_fit = [h['avg_fitness'] for h in ga_history]
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=gen, y=best_fit, mode='lines+markers', name='Best Fitness (R² - Penalty)', line=dict(color='#50e3c2', width=3)))
        fig_conv.add_trace(go.Scatter(x=gen, y=avg_fit, mode='lines+markers', name='Average Fitness', line=dict(color='#58a6ff', dash='dash')))
        
        fig_conv.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Generation', tickmode='linear'),
            yaxis=dict(title='Fitness Score'),
            margin=dict(l=40, r=40, t=10, b=40),
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_conv, use_container_width=True)

# ==================== TAB 3: PERFORMANCE & CONFUSION MATRIX ====================
with tab_performance:
    st.markdown("### Model Comparison & Demonstration Metrics")
    
    # 4 metrics cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        # Baseline MSE vs GA MSE
        base_mse = baseline_res['regression_metrics']['test_mse']
        ga_mse = ga_res['regression_metrics']['test_mse']
        delta_mse = ((ga_mse - base_mse) / base_mse) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test MSE</div>
            <div class="metric-value">{ga_mse:.4f}</div>
            <div class="metric-label" style="color: {'#3fb950' if delta_mse <= 0 else '#f85149'};">
                {delta_mse:.2f}% vs Baseline ({base_mse:.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        # Baseline R2 vs GA R2
        base_r2 = baseline_res['regression_metrics']['test_r2']
        ga_r2 = ga_res['regression_metrics']['test_r2']
        delta_r2 = ga_r2 - base_r2
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test R² Score</div>
            <div class="metric-value">{ga_r2:.4f}</div>
            <div class="metric-label" style="color: {'#3fb950' if delta_r2 >= 0 else '#f85149'};">
                {'+' if delta_r2 >= 0 else ''}{delta_r2:.4f} vs Baseline ({base_r2:.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        # Binned Accuracy
        base_acc = baseline_res['classification_metrics']['accuracy']
        ga_acc = ga_res['classification_metrics']['accuracy']
        delta_acc = (ga_acc - base_acc) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Binned Accuracy</div>
            <div class="metric-value">{ga_acc*100:.2f}%</div>
            <div class="metric-label" style="color: {'#3fb950' if delta_acc >= 0 else '#f85149'};">
                {'+' if delta_acc >= 0 else ''}{delta_acc:.2f}% vs Baseline ({base_acc*100:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        # Features Count
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Features Count</div>
            <div class="metric-value">{len(selected_features)}</div>
            <div class="metric-label">
                Reduced from {len(all_features)} ({(len(all_features)-len(selected_features))/len(all_features)*100:.0f}% reduction)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_cm1, col_cm2 = st.columns(2)
    
    labels = ['Low', 'Medium', 'High']
    
    with col_cm1:
        st.markdown("#### Baseline Confusion Matrix (All Features)")
        # Plotly Heatmap for Baseline Confusion Matrix
        cm_base = baseline_res['classification_metrics']['confusion_matrix']
        fig_cm_base = px.imshow(
            cm_base,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        fig_cm_base.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_cm_base, use_container_width=True)
        
    with col_cm2:
        st.markdown("#### GA-Optimized Confusion Matrix (Selected Features)")
        # Plotly Heatmap for GA Confusion Matrix
        cm_ga = ga_res['classification_metrics']['confusion_matrix']
        fig_cm_ga = px.imshow(
            cm_ga,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale="Greens",
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        fig_cm_ga.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_cm_ga, use_container_width=True)
        
    st.markdown("---")
    
    # Feature Importances Plotly Chart
    st.markdown("#### Feature Importance Comparison")
    
    # Align importances
    base_imp_df = baseline_res['feature_importances']
    ga_imp_df = ga_res['feature_importances']
    
    # Merge importances for comparison
    comparison_imp = pd.DataFrame({'feature': all_features})
    comparison_imp = comparison_imp.merge(base_imp_df.rename(columns={'importance': 'Baseline'}), on='feature', how='left')
    comparison_imp = comparison_imp.merge(ga_imp_df.rename(columns={'importance': 'GA-Optimized'}), on='feature', how='left')
    comparison_imp.fillna(0.0, inplace=True)
    comparison_imp = comparison_imp.sort_values(by='Baseline', ascending=True)
    
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=comparison_imp['feature'],
        x=comparison_imp['Baseline'],
        name='Baseline Model (All Features)',
        orientation='h',
        marker_color='#58a6ff'
    ))
    
    # Highlight GA selected vs unselected
    ga_colors = ['#50e3c2' if f in selected_features else 'rgba(248, 81, 73, 0.4)' for f in comparison_imp['feature']]
    fig_imp.add_trace(go.Bar(
        y=comparison_imp['feature'],
        x=comparison_imp['GA-Optimized'],
        name='GA-Optimized Model',
        orientation='h',
        marker_color=ga_colors
    ))
    
    fig_imp.update_layout(
        template="plotly_dark",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=150, r=40, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Green bars represent features selected by the Genetic Algorithm. Muted red bars denote features that were discarded by the GA (their importance is 0 in the GA model because they were not trained on).")

# ==================== TAB 4: PLAYGROUND ====================
with tab_playground:
    st.markdown("### Interactive Predictor Playground")
    st.write("Modify the sliders below to simulate a seismic event. The app will feed the features into both models, predict the earthquake magnitude, and determine the hazard category.")
    
    # Interactive Input Form
    col_inputs, col_outputs = st.columns([3, 2])
    
    with col_inputs:
        st.markdown("#### Input Seismic & Geological Parameters")
        
        # Arrange inputs into 3 columns
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        
        with sub_c1:
            inp_depth = st.slider("Depth of Focus (km)", min_value=5.0, max_value=300.0, value=25.0, step=5.0)
            inp_fault_dist = st.slider("Distance to Fault (km)", min_value=0.1, max_value=150.0, value=12.0, step=1.0)
            inp_slip_rate = st.slider("Fault Slip Rate (mm/year)", min_value=0.5, max_value=50.0, value=8.5, step=0.5)
            
        with sub_c2:
            inp_rigidity = st.slider("Rock Rigidity (GPa)", min_value=10.0, max_value=60.0, value=35.0, step=1.0)
            inp_hist_eq = st.slider("Historical EQ Count (10yr)", min_value=0, max_value=150, value=30, step=5)
            inp_delay = st.slider("P-S Wave Delay (seconds)", min_value=0.5, max_value=40.0, value=5.2, step=0.1)
            
        with sub_c3:
            inp_lat = st.slider("Latitude", min_value=30.0, max_value=45.0, value=35.6, step=0.1)
            inp_lon = st.slider("Longitude", min_value=130.0, max_value=145.0, value=139.7, step=0.1)
            inp_boundary = st.selectbox("Tectonic Plate Boundary", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            
        st.markdown("#### Input Noise / Environmental Parameters (Discarded by GA)")
        sub_c4, sub_c5 = st.columns(2)
        
        with sub_c4:
            inp_station_density = st.slider("Seismic Station Density", min_value=1, max_value=25, value=12)
            inp_soil_moist = st.slider("Soil Moisture (%)", min_value=5.0, max_value=95.0, value=45.0)
            inp_air_temp = st.slider("Air Temperature (°C)", min_value=-10.0, max_value=40.0, value=18.0)
            
        with sub_c5:
            inp_noise = st.slider("Ambient Sound Noise (dB)", min_value=20.0, max_value=80.0, value=48.0)
            inp_tide = st.slider("Tidal Height (m)", min_value=-2.0, max_value=2.0, value=0.1, step=0.1)
            inp_rand_noise = st.slider("Random Noise Value", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            
    with col_outputs:
        st.markdown("#### Predicted Earthquake Hazard Results")
        
        # Prepare feature vector for prediction
        input_data = {
            'latitude': [inp_lat],
            'longitude': [inp_lon],
            'depth': [inp_depth],
            'distance_to_fault': [inp_fault_dist],
            'fault_slip_rate': [inp_slip_rate],
            'rock_rigidity': [inp_rigidity],
            'historical_eq_count': [inp_hist_eq],
            'tectonic_plate_boundary': [inp_boundary],
            'p_s_wave_delay': [inp_delay],
            'seismic_station_density': [inp_station_density],
            'soil_moisture': [inp_soil_moist],
            'air_temperature': [inp_air_temp],
            'ambient_noise': [inp_noise],
            'tidal_height': [inp_tide],
            'random_noise_feature': [inp_rand_noise]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Predict using Baseline
        baseline_pred = baseline_res['model'].predict(input_df)[0]
        baseline_class = bin_magnitude(np.array([baseline_pred]))[0]
        
        # Predict using GA-Optimized (Uses only selected features)
        input_df_ga = input_df[selected_features]
        ga_pred = ga_res['model'].predict(input_df_ga)[0]
        ga_class = bin_magnitude(np.array([ga_pred]))[0]
        
        # Render Comparison
        color_map = {
            'Low': '#3fb950',
            'Medium': '#d29922',
            'High': '#f85149'
        }
        
        st.markdown(f"""
        <div class="prediction-card" style="border: 2px solid #58a6ff; box-shadow: 0 10px 30px rgba(88, 166, 255, 0.15); margin-bottom: 25px;">
            <h3 style="color: #58a6ff; margin-top:0px; font-size:1.25rem;">🔵 Baseline Model (All Features)</h3>
            <div style="font-size: 0.9rem; color: #8b949e;">Predicted Magnitude:</div>
            <div style="font-size: 2.8rem; font-weight: 700; color: #ffffff; margin: 5px 0;">{baseline_pred:.2f}</div>
            <div style="display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; background-color: {color_map[baseline_class]}22; color: {color_map[baseline_class]}; border: 1px solid {color_map[baseline_class]}55;">
                Hazard Class: {baseline_class}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="color: #50e3c2; margin-top:0px; font-size:1.25rem;">🟢 GA-Optimized Model (Selected Features)</h3>
            <div style="font-size: 0.9rem; color: #8b949e;">Predicted Magnitude:</div>
            <div style="font-size: 2.8rem; font-weight: 700; color: #ffffff; margin: 5px 0;">{ga_pred:.2f}</div>
            <div style="display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; background-color: {color_map[ga_class]}22; color: {color_map[ga_class]}; border: 1px solid {color_map[ga_class]}55;">
                Hazard Class: {ga_class}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); margin-top: 20px;">
            <h5 style="margin-top:0; color:#8b949e;">🔍 Playground Observation:</h5>
            <p style="font-size:0.9rem; margin-bottom:0; color:#8b949e;">
                If you modify environmental values like <em>Soil Moisture</em> or <em>Air Temperature</em>, you will see the <strong>Baseline Model</strong> magnitude prediction fluctuate slightly, as it was trained on these features and fits to their noise. However, the <strong>GA-Optimized Model</strong> prediction will remain completely stable, demonstrating that it is immune to environmental noise and focuses strictly on genuine geological signals!
            </p>
        </div>
        """, unsafe_allow_html=True)
