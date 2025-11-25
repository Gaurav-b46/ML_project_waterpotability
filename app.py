import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
    }
    h2 {
        color: #2563eb;
    }
    h3 {
        color: #3b82f6;
    }
    .potable {
        color: #22c55e;
        font-size: 28px;
        font-weight: bold;
    }
    .not-potable {
        color: #ef4444;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD PICKLE FILES
# ============================================
@st.cache_resource
def load_model_files():
    """Load all pickle files"""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
        
        with open('feature_stats.pkl', 'rb') as file:
            feature_stats = pickle.load(file)
        
        with open('model_info.pkl', 'rb') as file:
            model_info = pickle.load(file)
        
        with open('dataframe.pkl', 'rb') as file:
            df = pickle.load(file)
        
        return model, feature_names, feature_stats, model_info, df
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

# Load all files
model, feature_names, feature_stats, model_info, df = load_model_files()

# ============================================
# HELPER FUNCTIONS
# ============================================
def save_prediction_history(features, prediction, probability):
    """Save prediction to history"""
    history_file = 'prediction_history.json'
    
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'features': features,
        'prediction': int(prediction),
        'probability': float(probability)
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(history_entry)
    
    # Keep only last 50 predictions
    history = history[-50:]
    
    with open(history_file, 'w') as f:
        json.dump(history, f)

def load_prediction_history():
    """Load prediction history"""
    history_file = 'prediction_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def create_gauge_chart(probability, prediction):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Potability Confidence", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_bar(probabilities):
    """Create probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Not Potable', 'Potable'],
            y=probabilities * 100,
            marker_color=['#ef4444', '#22c55e'],
            text=[f'{p:.2f}%' for p in probabilities * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability (%)",
        height=300,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_feature_comparison(input_values, feature_names, feature_stats):
    """Create feature comparison chart"""
    features = []
    input_vals = []
    mean_vals = []
    
    for i, name in enumerate(feature_names):
        features.append(name)
        input_vals.append(input_values[i])
        mean_vals.append(feature_stats[name]['mean'])
    
    fig = go.Figure(data=[
        go.Bar(name='Your Input', x=features, y=input_vals, marker_color='#3b82f6'),
        go.Bar(name='Dataset Mean', x=features, y=mean_vals, marker_color='#94a3b8')
    ])
    
    fig.update_layout(
        title="Your Input vs Dataset Average",
        xaxis_title="Features",
        yaxis_title="Value",
        barmode='group',
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart"""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in indices],
                y=[importances[i] for i in indices],
                marker_color='#8b5cf6',
                text=[f'{importances[i]:.3f}' for i in indices],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    except:
        return None

def create_history_chart(history):
    """Create prediction history chart"""
    if not history:
        return None
    
    df_history = pd.DataFrame(history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_history['timestamp'],
        y=df_history['probability'],
        mode='lines+markers',
        name='Prediction Probability',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis_title="Potability Probability",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("<h1>💧 Water Potability Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if model is loaded
    if model is None:
        st.error("⚠️ Failed to load model files. Please ensure all pickle files are present.")
        return
    
    # Sidebar - Model Information
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/water.png", width=100)
        st.markdown("## 📊 Model Information")
        st.info(f"""
        **Best Parameters:**
        - Criterion: {model_info['best_params']['criterion']}
        - Splitter: {model_info['best_params']['splitter']}
        - Min Samples Split: {model_info['best_params']['min_samples_split']}
        
        **Performance:**
        - CV Score: {model_info['best_score']:.4f}
        - Training Score: {model_info['training_score']:.4f}
        - Testing Score: {model_info['testing_score']:.4f}
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 About")
        st.write("""
        This app predicts whether water is safe to drink based on various water quality parameters.
        
        **Features:**
        - Real-time predictions
        - Interactive visualizations
        - Prediction history
        - Feature importance analysis
        """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Analytics", "📜 History", "ℹ️ Dataset Info"])
    
    # ============================================
    # TAB 1: PREDICTION
    # ============================================
    with tab1:
        st.markdown("### Enter Water Quality Parameters")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Sliders", "Number Input", "Random Sample"],
            horizontal=True
        )
        
        input_values = []
        
        if input_method == "Random Sample":
            if st.button("🎲 Generate Random Sample", key="random"):
                st.session_state.random_sample = True
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        columns = [col1, col2, col3]
        
        for i, feature in enumerate(feature_names):
            with columns[i % 3]:
                stats = feature_stats[feature]
                
                if input_method == "Sliders":
                    if 'random_sample' in st.session_state and st.session_state.random_sample:
                        default_val = np.random.uniform(stats['min'], stats['max'])
                    else:
                        default_val = stats['mean']
                    
                    value = st.slider(
                        f"**{feature}**",
                        min_value=float(stats['min']),
                        max_value=float(stats['max']),
                        value=float(default_val),
                        key=f"slider_{feature}"
                    )
                elif input_method == "Number Input":
                    value = st.number_input(
                        f"**{feature}**",
                        min_value=float(stats['min']),
                        max_value=float(stats['max']),
                        value=float(stats['mean']),
                        key=f"number_{feature}"
                    )
                else:  # Random Sample
                    if 'random_sample' in st.session_state and st.session_state.random_sample:
                        value = np.random.uniform(stats['min'], stats['max'])
                    else:
                        value = stats['mean']
                    st.metric(f"**{feature}**", f"{value:.2f}")
                
                input_values.append(value)
                st.caption(f"Range: {stats['min']:.2f} - {stats['max']:.2f}")
        
        # Reset random sample flag
        if 'random_sample' in st.session_state:
            st.session_state.random_sample = False
        
        st.markdown("---")
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 PREDICT WATER POTABILITY", key="predict")
        
        # Make prediction
        if predict_button:
            input_array = np.array([input_values])
            prediction = model.predict(input_array)[0]
            probabilities = model.predict_proba(input_array)[0]
            
            # Save to history
            save_prediction_history(input_values, prediction, probabilities[1])
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Results columns
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.markdown('<p class="potable">✅ WATER IS POTABLE</p>', unsafe_allow_html=True)
                    st.success("This water is safe to drink!")
                else:
                    st.markdown('<p class="not-potable">❌ WATER IS NOT POTABLE</p>', unsafe_allow_html=True)
                    st.error("This water is NOT safe to drink!")
                
                st.metric("Confidence", f"{max(probabilities)*100:.2f}%")
                st.metric("Potable Probability", f"{probabilities[1]*100:.2f}%")
                st.metric("Not Potable Probability", f"{probabilities[0]*100:.2f}%")
            
            with result_col2:
                st.plotly_chart(create_gauge_chart(probabilities[1], prediction), use_container_width=True)
            
            # Probability bar chart
            st.plotly_chart(create_probability_bar(probabilities), use_container_width=True)
            
            # Feature comparison
            st.plotly_chart(create_feature_comparison(input_values, feature_names, feature_stats), use_container_width=True)
    
    # ============================================
    # TAB 2: ANALYTICS
    # ============================================
    with tab2:
        st.markdown("### 📊 Model Analytics")
        
        # Feature importance
        importance_fig = create_feature_importance_chart(model, feature_names)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Distribution plots
        st.markdown("### 📉 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_select1 = st.selectbox("Select Feature 1:", feature_names, key="feat1")
            fig1 = px.histogram(df, x=feature_select1, color='Potability',
                               title=f'Distribution of {feature_select1}',
                               color_discrete_map={0: '#ef4444', 1: '#22c55e'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            feature_select2 = st.selectbox("Select Feature 2:", feature_names, index=1, key="feat2")
            fig2 = px.histogram(df, x=feature_select2, color='Potability',
                               title=f'Distribution of {feature_select2}',
                               color_discrete_map={0: '#ef4444', 1: '#22c55e'})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 🔍 Feature Relationship Analysis")
        scatter_col1, scatter_col2 = st.columns(2)
        
        with scatter_col1:
            x_feature = st.selectbox("X-axis:", feature_names, key="x_feat")
        with scatter_col2:
            y_feature = st.selectbox("Y-axis:", feature_names, index=1, key="y_feat")
        
        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, color='Potability',
                                title=f'{x_feature} vs {y_feature}',
                                color_discrete_map={0: '#ef4444', 1: '#22c55e'})
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### 🔥 Correlation Heatmap")
        corr_matrix = df[feature_names].corr()
        fig_corr = px.imshow(corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # ============================================
    # TAB 3: HISTORY
    # ============================================
    with tab3:
        st.markdown("### 📜 Prediction History")
        
        history = load_prediction_history()
        
        if history:
            # History chart
            history_fig = create_history_chart(history)
            if history_fig:
                st.plotly_chart(history_fig, use_container_width=True)
            
            # History table
            st.markdown("### Recent Predictions")
            
            history_df = pd.DataFrame(history)
            history_df['prediction_label'] = history_df['prediction'].map({0: '❌ Not Potable', 1: '✅ Potable'})
            history_df['probability'] = history_df['probability'].apply(lambda x: f"{x*100:.2f}%")
            
            display_df = history_df[['timestamp', 'prediction_label', 'probability']].tail(10)
            display_df.columns = ['Timestamp', 'Prediction', 'Confidence']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                if os.path.exists('prediction_history.json'):
                    os.remove('prediction_history.json')
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No prediction history yet. Make some predictions to see them here!")
    
    # ============================================
    # TAB 4: DATASET INFO
    # ============================================
    with tab4:
        st.markdown("### 📋 Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Potable Samples", len(df[df['Potability'] == 1]))
        with col3:
            st.metric("Not Potable Samples", len(df[df['Potability'] == 0]))
        
        # Dataset statistics
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Class distribution
        st.markdown("### 🥧 Class Distribution")
        class_counts = df['Potability'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Not Potable', 'Potable'],
            values=class_counts.values,
            marker_colors=['#ef4444', '#22c55e'],
            hole=0.3
        )])
        fig_pie.update_layout(title="Water Potability Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature descriptions
        st.markdown("### 📖 Feature Descriptions")
        feature_descriptions = {
            'ph': 'pH level of water (0-14)',
            'Hardness': 'Water hardness, capacity of water to precipitate soap in mg/L',
            'Solids': 'Total dissolved solids in ppm',
            'Chloramines': 'Amount of Chloramines in ppm',
            'Sulfate': 'Amount of Sulfates dissolved in mg/L',
            'Conductivity': 'Electrical conductivity of water in μS/cm',
            'Organic_carbon': 'Amount of organic carbon in ppm',
            'Trihalomethanes': 'Amount of Trihalomethanes in μg/L',
            'Turbidity': 'Measure of light emitting property of water in NTU'
        }
        
        for feature, description in feature_descriptions.items():
            if feature in feature_names:
                st.markdown(f"**{feature}:** {description}")

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()