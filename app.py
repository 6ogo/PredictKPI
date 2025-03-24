"""
Multi-KPI Predictor Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import joblib
from dotenv import load_dotenv
import logging
import traceback

# Load custom modules
from utils import (
    logger, BOLAG_VALUES, DIALOG_VALUES, SYFTE_VALUES, PRODUKT_VALUES,
    get_current_model_version, list_available_model_versions, get_available_kpi_models,
    load_and_prepare_data, send_to_groq_api, import_subjects_from_csv, export_results_to_csv
)
from enhanced_features import (
    prepare_multi_kpi_features, create_campaign_level_features
)
from multi_kpi_models import (
    MultiKpiModelManager, OptimalContentSelector
)
from ui_components import (
    create_enhanced_input_ui, display_multi_kpi_results, create_kpi_dashboard,
    create_batch_prediction_ui, create_model_comparison_ui
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Setup Streamlit page
st.set_page_config(
    page_title="Multi-KPI Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {margin-bottom: 1rem;}
    h2 {margin-top: 1.5rem;}
    .stMetric {border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: 0.5rem;}
    .css-1r6slb0 {border-width: 1px; border-style: solid; border-radius: 0.5rem;}
    .block-container {max-width: 1200px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(delivery_file='Data/delivery_data.csv', customer_file='Data/customer_data.csv'):
    """
    Load delivery and customer data with error handling and caching
    """
    try:
        return load_and_prepare_data(delivery_file, customer_file)
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        raise

@st.cache_resource
def get_model_manager():
    """
    Get the model manager as a cached resource
    """
    return MultiKpiModelManager()

def main():
    st.title('Multi-KPI Email Campaign Optimizer')
    
    # Sidebar for model version selection
    st.sidebar.header("Model Settings")
    available_versions = list_available_model_versions()
    
    if available_versions:
        selected_version = st.sidebar.selectbox(
            "Select Model Version",
            options=available_versions,
            index=len(available_versions) - 1  # Default to latest version
        )
    else:
        selected_version = get_current_model_version()
        st.sidebar.info(f"No saved models found. Will train version {selected_version} if needed.")
    
    # Show available KPI models for selected version
    available_kpi_models = get_available_kpi_models(selected_version)
    if selected_version in available_kpi_models:
        kpi_types = available_kpi_models[selected_version]
        if kpi_types:
            st.sidebar.success(f"Available KPI models: {', '.join(kpi_types)}")
        else:
            st.sidebar.warning(f"No KPI models found for version {selected_version}")
    
    # Force retrain option
    force_retrain = st.sidebar.checkbox("Force model retraining")
    
    # Select KPI models to train/use
    kpi_types = st.sidebar.multiselect(
        "KPI Types to Use",
        options=["openrate", "clickrate", "optoutrate"],
        default=["openrate", "clickrate", "optoutrate"]
    )
    
    # Model types to try
    model_types = st.sidebar.multiselect(
        "Model Types to Try",
        options=["xgboost", "lightgbm", "catboost"],
        default=["xgboost", "lightgbm"]
    )
    
    # Load data
    try:
        delivery_data, customer_data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Campaign Optimizer', 'Batch Processing', 'Model Results', 'KPI Dashboard'])
    
    # Initialize model manager
    model_manager = get_model_manager()
    model_manager.kpi_types = kpi_types
    
    # Prepare features for models
    try:
        features_dict, target_dict = prepare_multi_kpi_features(delivery_data, customer_data)
        
        # Train or load models
        trained_models = model_manager.train_all_kpi_models(
            features_dict, 
            target_dict,
            model_types=model_types,
            force_retrain=force_retrain,
            version=selected_version
        )
        
        # Check if training was successful
        if not trained_models:
            st.warning("No models were successfully trained. Check logs for details.")
        else:
            # Show best model info
            best_model_info = model_manager.get_best_model_info()
            st.sidebar.subheader("Best Models Selected")
            
            for kpi_type, info in best_model_info.items():
                model_type = info.get('model_type', 'unknown')
                performance = info.get('performance', {})
                r2 = performance.get('r2', 0)
                st.sidebar.info(f"{kpi_type}: {model_type.upper()} (R² = {r2:.4f})")
                
    except Exception as e:
        st.error(f"Error preparing features or training models: {str(e)}")
        st.error(traceback.format_exc())
        return
    
    # Tab 1: Campaign Optimizer
    with tab1:
        st.header('Optimize Email Campaign')
        st.info(f"Using model version {selected_version}")
        
        # UI for campaign settings and content
        dialog_options = sorted(DIALOG_VALUES.items())
        dialog_labels = [(code[0], label) for key, (code, label) in dialog_options]
        
        selected_dialog_display = st.selectbox(
            'Dialog',
            options=[label for _, label in dialog_labels]
        )
        selected_dialog_code = next(code for code, label in dialog_labels if label == selected_dialog_display)
        
        syfte_options = sorted(SYFTE_VALUES.items())
        syfte_labels = [(code[0], label) for key, (code, label) in syfte_options]
        
        selected_syfte_display = st.selectbox(
            'Syfte',
            options=[label for _, label in syfte_labels]
        )
        selected_syfte_code = next(code for code, label in syfte_labels if label == selected_syfte_display)
        
        product_options = sorted(PRODUKT_VALUES.items())
        product_labels = [(code[0], label) for key, (code, label) in product_options]
        
        selected_product_display = st.selectbox(
            'Product',
            options=[label for _, label in product_labels]
        )
        selected_product_code = next(code for code, label in product_labels if label == selected_product_display)
        
        # Audience selection
        st.subheader("Audience Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
        
        with col2:
            max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
        
        bolag_options = sorted(BOLAG_VALUES.keys())
        excluded_bolag_display = st.multiselect('Exclude Bolag', bolag_options)
        included_bolag = [b for b in bolag_options if b not in excluded_bolag_display]
        
        # Email content
        st.subheader("Email Content")
        subject_line, preheader, use_genai, genai_options = create_enhanced_input_ui()
        
        # Campaign metadata
        campaign_metadata = {
            'dialog': selected_dialog_code,
            'dialog_display': selected_dialog_display,
            'syfte': selected_syfte_code,
            'syfte_display': selected_syfte_display,
            'product': selected_product_code,
            'product_display': selected_product_display,
            'min_age': min_age,
            'max_age': max_age,
            'included_bolag': included_bolag
        }
        
        # Prediction logic
        if subject_line and preheader:
            try:
                # Create features for this campaign
                campaign_features = create_campaign_level_features(
                    subject_line, preheader, campaign_metadata
                )
                
                # Make predictions
                predictions = model_manager.predict_new_campaign(
                    subject_line, preheader, campaign_metadata, selected_version
                )
                
                # Display predicted results
                st.subheader('Predicted Results')
                col1, col2, col3 = st.columns(3)
                
                openrate = predictions.get('openrate', 0)
                clickrate = predictions.get('clickrate', 0)
                optoutrate = predictions.get('optoutrate', 0)
                
                col1.metric("Open Rate", f"{openrate:.2%}")
                col2.metric("Click Rate", f"{clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{optoutrate:.2%}")
                
                # A/B/C/D Testing with Groq API
                if use_genai and GROQ_API_KEY:
                    if st.button('Generate Alternatives with AI'):
                        with st.spinner("Generating alternatives..."):
                            # Send to API
                            response_data = send_to_groq_api(
                                subject_line, preheader, 
                                predictions,
                                campaign_metadata,
                                GROQ_API_KEY
                            )
                            
                            if "error" in response_data:
                                st.error(response_data["error"])
                                if "raw_content" in response_data:
                                    with st.expander("Raw API Response"):
                                        st.code(response_data["raw_content"])
                            else:
                                try:
                                    suggestions = response_data.get('suggestions', [])
                                    
                                    if suggestions:
                                        # Create content selector
                                        content_selector = OptimalContentSelector(model_manager)
                                        
                                        # Evaluate alternatives
                                        optimization_results = content_selector.optimize_content(
                                            subject_line, 
                                            preheader,
                                            suggestions,
                                            campaign_metadata,
                                            selected_version
                                        )
                                        
                                        # Display results
                                        display_multi_kpi_results(
                                            optimization_results['options'],
                                            optimization_results['best_option'],
                                            optimization_results['improvement']
                                        )
                                    else:
                                        st.warning("No suggestions received from API.")
                                        
                                except Exception as e:
                                    st.error(f"Error processing alternatives: {str(e)}")
                                    logger.error(f"Error processing alternatives: {str(e)}")
                                    logger.error(traceback.format_exc())
                elif use_genai and not GROQ_API_KEY:
                    st.error("Groq API key not found. Please set GROQ_API_KEY in .env file.")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                logger.error(f"Error making predictions: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Tab 2: Batch Processing
    with tab2:
        try:
            create_batch_prediction_ui(model_manager, selected_version)
        except Exception as e:
            st.error(f"Error in batch prediction UI: {str(e)}")
            logger.error(f"Error in batch prediction UI: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Tab 3: Model Results
    with tab3:
        try:
            create_model_comparison_ui(
                model_manager, 
                delivery_data, 
                customer_data,
                selected_version
            )
        except Exception as e:
            st.error(f"Error in model comparison UI: {str(e)}")
            logger.error(f"Error in model comparison UI: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Tab 4: KPI Dashboard
    with tab4:
        try:
            create_kpi_dashboard(delivery_data)
        except Exception as e:
            st.error(f"Error creating KPI dashboard: {str(e)}")
            logger.error(f"Error creating KPI dashboard: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()