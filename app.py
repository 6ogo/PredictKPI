"""
Main application file for KPI Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import logging
import traceback

# Load custom modules
from utils import (
    logger, BOLAG_VALUES, DIALOG_VALUES, SYFTE_VALUES, PRODUKT_VALUES,
    get_current_model_version, get_model_filename, get_model_metadata_filename,
    save_model_metadata, load_model_metadata, list_available_models,
    categorize_age, send_to_groq_api
)
from features import (
    load_nlp_models, engineer_features, enhanced_feature_engineering,
    process_data_for_age_heatmap, prepare_version_heatmap_data,
    select_features, adapt_features_to_model
)
from models import (
    train_model, train_model_with_params, validate_model_features,
    evaluate_model, cross_validate_model, cross_validate_model_parallel,
    tune_hyperparameters, generate_model_documentation,
    generate_feature_importance_plot, generate_prediction_vs_actual_plot,
    generate_error_distribution_plot, extract_feature_importance,
    ModelFactory
)
from ui_components import (
    create_enhanced_input_ui, create_interactive_heatmap, create_age_heatmap,
    display_abcd_results, create_kpi_dashboard, create_batch_prediction_ui
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Setup Streamlit page
st.set_page_config(
    page_title="Sendout KPI Predictor",
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

# Cache data loading
@st.cache_data(ttl=3600)
def load_data(delivery_file='../Data/delivery_data.csv', customer_file='../Data/customer_data.csv'):
    """
    Load delivery and customer data with error handling and caching
    """
    try:
        logger.info(f"Loading data from {delivery_file} and {customer_file}")
        delivery_data = pd.read_csv(delivery_file, sep=';', encoding='utf-8')
        customer_data = pd.read_csv(customer_file, sep=';', encoding='utf-8')
        
        # Data validation
        required_delivery_cols = ['Subject', 'Preheader', 'Dialog', 'Syfte', 'Product', 'Opens', 'Sendouts', 'Clicks', 'Optouts', 'InternalName']
        required_customer_cols = ['InternalName', 'Age', 'Bolag']
        
        missing_delivery_cols = [col for col in required_delivery_cols if col not in delivery_data.columns]
        missing_customer_cols = [col for col in required_customer_cols if col not in customer_data.columns]
        
        if missing_delivery_cols:
            raise ValueError(f"Missing required columns in delivery data: {missing_delivery_cols}")
        if missing_customer_cols:
            raise ValueError(f"Missing required columns in customer data: {missing_customer_cols}")
            
        logger.info(f"Successfully loaded data - Delivery: {delivery_data.shape}, Customer: {customer_data.shape}")
        return delivery_data, customer_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        st.error(f"Data file not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        st.error(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error loading data: {str(e)}")
        raise

# Cache feature engineering
@st.cache_data(ttl=3600)
def engineer_features_cached(delivery_data, customer_data, include_preheader=True, enhanced=False):
    """Cached wrapper for the feature engineering function"""
    if enhanced:
        return enhanced_feature_engineering(delivery_data, customer_data, include_preheader)
    else:
        return engineer_features(delivery_data, customer_data, include_preheader)

# Cache model loading
@st.cache_resource
def load_model_cached(model_file):
    """Cached model loading to avoid repeatedly loading large models"""
    try:
        logger.info(f"Loading model from {model_file}")
        model = joblib.load(model_file)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def main():
    st.title('Email Sendout KPI Predictor')
    
    # Sidebar for model version selection
    st.sidebar.header("Model Settings")
    available_models = list_available_models()
    
    if available_models:
        selected_version = st.sidebar.selectbox(
            "Select Model Version",
            options=available_models,
            index=len(available_models) - 1  # Default to latest version
        )
    else:
        selected_version = get_current_model_version()
        st.sidebar.info(f"No saved models found. Will train version {selected_version} if needed.")
    
    model_file = get_model_filename(selected_version)
    metadata_file = get_model_metadata_filename(selected_version)
    
    # Force retrain option
    force_retrain = st.sidebar.checkbox("Force model retraining")
    
    # Enhanced options
    enable_enhanced_features = st.sidebar.checkbox("Use enhanced features", value=False,
                                                 help="Enable TF-IDF and interaction features")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        options=["xgboost", "lightgbm", "catboost", "ensemble", "nn"],
        index=0,
        help="Select model type (only applies when training new model)"
    )
    
    # Load data
    try:
        delivery_data, customer_data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # Engineer features
    try:
        # For v2.0.0+ include preheader features
        include_preheader = float(selected_version.split('.')[0]) >= 2
        features_dict, target, feature_metadata = engineer_features_cached(
            delivery_data, customer_data, 
            include_preheader=include_preheader,
            enhanced=enable_enhanced_features
        )
    except Exception as e:
        st.error(f"Failed to engineer features: {str(e)}")
        st.error(traceback.format_exc())
        return
    
    # Select appropriate feature set based on version
    if float(selected_version.split('.')[0]) >= 2:
        features = features_dict['v2']
        feature_set_key = 'v2'
    else:
        features = features_dict['legacy']
        feature_set_key = 'legacy'
    
    # Feature selection if needed
    if enable_enhanced_features and features.shape[1] > 300:
        st.sidebar.info(f"Using feature selection to reduce from {features.shape[1]} features")
        features, selected_feature_names = select_features(features, target, max_features=300)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Define default sample weights
    THRESHOLD = 0.5
    WEIGHT_HIGH = 2.0
    WEIGHT_LOW = 1.0
    sample_weights_train = np.where(y_train > THRESHOLD, WEIGHT_HIGH, WEIGHT_LOW)
    
    # Load or train model
    model_loaded = False
    if os.path.exists(model_file) and not force_retrain:
        try:
            model = load_model_cached(model_file)
            logger.info(f"Loaded model from {model_file}")
            model_loaded = True
            
            # Validate features
            valid, message = validate_model_features(model, features)
            if not valid:
                st.warning(f"Feature mismatch detected: {message}")
                st.info("Adapting features to match model expectations...")
                
                # Adapt features to match model
                features_adapted = adapt_features_to_model(model, features)
                X_train_adapted, X_test_adapted, _, _ = train_test_split(features_adapted, target, test_size=0.2, random_state=42)
                
                # Use adapted features for prediction
                X_train, X_test = X_train_adapted, X_test_adapted
            
            metadata = load_model_metadata(selected_version)
            if metadata:
                st.sidebar.info(f"Model v{selected_version} loaded with {len(model.feature_names_) if hasattr(model, 'feature_names_') else 'unknown'} features")
                st.sidebar.info(f"Trained on {metadata.get('training_samples', 'unknown')} samples")
            else:
                st.sidebar.info(f"Model v{selected_version} loaded but no metadata found")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Training new model instead...")
            force_retrain = True
    
    if force_retrain or not os.path.exists(model_file):
        try:
            logger.info(f"Training new model version {selected_version}")
            st.info(f"Training new model version {selected_version}...")
            
            # Train model with default parameters
            params = {
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            }
            
            model = train_model(
                X_train, y_train, 
                sample_weights=sample_weights_train, 
                params=params,
                model_type=model_type
            )
            
            # Save model
            joblib.dump(model, model_file)
            logger.info(f"Saved model to {model_file}")
            
            # Save metadata
            metadata = {
                'version': selected_version,
                'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'feature_set': feature_set_key,
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_names': X_train.columns.tolist(),
                'feature_count': X_train.shape[1],
                'include_preheader': include_preheader,
                'enhanced_features': enable_enhanced_features,
                'model_type': model_type,
                'model_parameters': params,
                'sample_weights': {
                    'threshold': THRESHOLD,
                    'weight_high': WEIGHT_HIGH,
                    'weight_low': WEIGHT_LOW
                }
            }
            save_model_metadata(metadata, selected_version)
            
            st.success(f"Trained and saved new model version {selected_version}")
            model_loaded = True
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.error(traceback.format_exc())
            return
    
    # Initialize variables with default values to prevent UnboundLocalError
    test_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
    y_pred_test = None
    cv_results = {
        'mse': {'mean': 0, 'std': 0, 'scores': []},
        'rmse': {'mean': 0, 'std': 0, 'scores': []},
        'mae': {'mean': 0, 'std': 0, 'scores': []},
        'r2': {'mean': 0, 'std': 0, 'scores': []}
    }
    full_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
    y_pred_full = None
    
    # Evaluate model
    try:
        # Adapt features to match model's expected features
        if hasattr(model, 'feature_names_'):
            logger.info("Adapting features to match model expectations")
            X_train_adapted = adapt_features_to_model(model, X_train)
            X_test_adapted = adapt_features_to_model(model, X_test)
            features_adapted = adapt_features_to_model(model, features)
            
            # Use adapted features
            X_train, X_test = X_train_adapted, X_test_adapted
            features_for_prediction = features_adapted
        else:
            features_for_prediction = features
        
        # Cross-validation - use parallel version if available
        try:
            cv_results = cross_validate_model_parallel(
                X_train, y_train, 
                params={'reg_lambda': 1.0, 'random_state': 42},
                sample_weights=sample_weights_train
            )
        except Exception as e:
            logger.warning(f"Parallel cross-validation failed: {str(e)}. Using serial version.")
            cv_results = cross_validate_model(
                X_train, y_train, 
                params={'reg_lambda': 1.0, 'random_state': 42},
                sample_weights=sample_weights_train
            )
        
        # Test set evaluation
        test_metrics, y_pred_test = evaluate_model(model, X_test, y_test)
        
        # Full dataset verification
        full_metrics, y_pred_full = evaluate_model(model, features_for_prediction, target)
        
        # Generate documentation
        if not os.path.exists(os.path.join('Docs', f"model_v{selected_version}", 'model_documentation.yaml')):
            generate_model_documentation(
                model, feature_metadata, 
                {'train_samples': X_train.shape[0]},
                cv_results, test_metrics, 
                version=selected_version
            )
            
            # Generate plots
            if hasattr(model, 'feature_names_'):
                generate_feature_importance_plot(model, model.feature_names_, version=selected_version)
            else:
                generate_feature_importance_plot(model, features.columns, version=selected_version)
                
            generate_prediction_vs_actual_plot(y_test, y_pred_test, version=selected_version)
            generate_error_distribution_plot(y_test, y_pred_test, version=selected_version)
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        logger.error(f"Error evaluating model: {str(e)}")
        logger.error(traceback.format_exc())
        st.warning("Using default metrics due to evaluation error. Some visualizations may not be available.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Sendout Prediction', 'Batch Processing', 'Model Results', 'KPI Dashboard'])
    
    # Tab 3: Model Performance
    with tab3:
        st.header('Model Performance')
        st.subheader(f"Model Version: {selected_version}")
        
        # Add retraining section with parameters
        with st.expander("Retrain Model with Custom Parameters", expanded=False):
            st.write("Adjust model parameters and click 'Retrain Model' to create a new version.")
            
            # Model type selection
            retrain_model_type = st.selectbox(
                "Model Type",
                options=["xgboost", "lightgbm", "catboost", "ensemble", "nn"],
                index=0,
                help="Select model type for retraining"
            )
            
            # NN-specific parameters
            nn_params = {}
            if retrain_model_type == 'nn':
                with st.expander("Neural Network Parameters", expanded=False):
                    nn_params['learning_rate'] = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, 
                                                         format="%.4f")
                    nn_params['epochs'] = st.slider("Max Epochs", 50, 500, 100, 10)
                    nn_params['batch_size'] = st.slider("Batch Size", 16, 128, 32, 8)
                    nn_params['dropout_rate'] = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
                    
                    # Hidden layers
                    hidden_layers_str = st.text_input("Hidden Layers (comma-separated)",
                                                    value="64,32",
                                                    help="Comma-separated list of neurons in each hidden layer")
                    try:
                        nn_params['hidden_layers'] = [int(x) for x in hidden_layers_str.split(',')]
                    except:
                        st.warning("Invalid hidden layers format. Using default [64, 32].")
                        nn_params['hidden_layers'] = [64, 32]
            
            col1, col2 = st.columns(2)
            with col1:
                # XGBoost parameters
                st.subheader("Model Parameters")
                reg_lambda = st.slider("L2 Regularization (reg_lambda)", 0.01, 10.0, 1.0, 0.1, 
                                    help="Higher values increase regularization strength to prevent overfitting")
                n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 10, 
                                        help="Number of boosting rounds/trees")
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, 
                                        help="Step size for each boosting round")
                max_depth = st.slider("Max Tree Depth", 3, 10, 5, 1, 
                                    help="Maximum depth of each tree")
            
            with col2:
                # Sample weight parameters
                st.subheader("Sample Weight Configuration")
                sw_enabled = st.checkbox("Enable Sample Weights", value=True, 
                                        help="Give different weights to samples based on their target value")
                if sw_enabled:
                    sw_threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, 
                                            help="Samples with target above this threshold get higher weight")
                    sw_weight_high = st.slider("High Weight", 1.1, 5.0, 2.0, 0.1, 
                                            help="Weight for samples above threshold")
                    sw_weight_low = st.slider("Low Weight", 0.5, 1.0, 1.0, 0.05, 
                                            help="Weight for samples below threshold")
                else:
                    sw_threshold = 0.7
                    sw_weight_high = 1.0
                    sw_weight_low = 1.0
            
            # Version options
            st.subheader("Version Control")
            use_today_version = st.checkbox("Use today's date for version", value=True, 
                                            help="Creates a version based on today's date (YY.MM.DD)")
            if not use_today_version:
                custom_version = st.text_input("Custom Version", value=selected_version, 
                                            help="Specify a custom version number (e.g., '2.1.0')")
            
            # Enhanced features
            retrain_enhanced_features = st.checkbox("Use enhanced features for retraining", value=enable_enhanced_features,
                                                  help="Include TF-IDF and interaction features")
            
            # Retrain button
            if st.button("Retrain Model"):
                try:
                    # Set version
                    new_version = get_current_model_version() if use_today_version else custom_version
                    
                    # Show training status
                    with st.spinner(f"Training model version {new_version}..."):
                        # Get enhanced features if requested
                        if retrain_enhanced_features != enable_enhanced_features:
                            # Re-engineer features with new settings
                            enhanced_features_dict, enhanced_target, enhanced_feature_metadata = enhanced_feature_engineering(
                                delivery_data, customer_data, 
                                include_preheader=include_preheader
                            )
                            
                            # Select feature set
                            if float(new_version.split('.')[0]) >= 2:
                                enhanced_features = enhanced_features_dict['v2']
                                enhanced_feature_set_key = 'v2'
                            else:
                                enhanced_features = enhanced_features_dict['legacy']
                                enhanced_feature_set_key = 'legacy'
                                
                            # Feature selection if needed
                            if enhanced_features.shape[1] > 300:
                                enhanced_features, selected_features_names = select_features(
                                    enhanced_features, enhanced_target, max_features=300
                                )
                                
                            # Split enhanced data
                            enhanced_X_train, enhanced_X_test, enhanced_y_train, enhanced_y_test = train_test_split(
                                enhanced_features, enhanced_target, test_size=0.2, random_state=42
                            )
                            
                            # Use enhanced features for training
                            X_train, X_test, y_train, y_test = enhanced_X_train, enhanced_X_test, enhanced_y_train, enhanced_y_test
                            features = enhanced_features
                            feature_set_key = enhanced_feature_set_key
                            feature_metadata = enhanced_feature_metadata
                        
                        # Configure parameters
                        if retrain_model_type == 'nn':
                            # Neural network parameters
                            model_params = nn_params
                        else:
                            # Tree-based model parameters
                            model_params = {
                                'reg_lambda': reg_lambda,
                                'n_estimators': n_estimators,
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'random_state': 42
                            }
                        
                        # Configure sample weights
                        if sw_enabled:
                            sample_weight_config = {
                                'threshold': sw_threshold,
                                'weight_high': sw_weight_high,
                                'weight_low': sw_weight_low
                            }
                            sample_weights = np.where(y_train > sw_threshold, sw_weight_high, sw_weight_low)
                        else:
                            sample_weight_config = None
                            sample_weights = None
                        
                        # Train new model
                        new_model = train_model_with_params(
                            X_train, y_train, 
                            params=model_params,
                            sample_weight_config=sample_weight_config,
                            model_type=retrain_model_type
                        )
                        
                        # Save model
                        new_model_file = get_model_filename(new_version)
                        joblib.dump(new_model, new_model_file)
                        
                        # Evaluate new model
                        test_metrics, y_pred_test = evaluate_model(new_model, X_test, y_test)
                        
                        # Save metadata
                        metadata = {
                            'version': new_version,
                            'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'feature_set': feature_set_key,
                            'training_samples': X_train.shape[0],
                            'test_samples': X_test.shape[0],
                            'feature_names': X_train.columns.tolist(),
                            'feature_count': X_train.shape[1],
                            'include_preheader': include_preheader,
                            'enhanced_features': retrain_enhanced_features,
                            'model_type': retrain_model_type,
                            'model_parameters': model_params,
                            'sample_weights': sample_weight_config
                        }
                        save_model_metadata(metadata, new_version)
                        
                        # Generate documentation
                        cv_results = cross_validate_model(
                            X_train, y_train, 
                            params=model_params,
                            sample_weights=sample_weights
                        )
                        
                        generate_model_documentation(
                            new_model, feature_metadata, 
                            {'train_samples': X_train.shape[0]},
                            cv_results, test_metrics, 
                            version=new_version
                        )
                        
                        # Generate plots
                        if hasattr(new_model, 'feature_names_'):
                            generate_feature_importance_plot(new_model, new_model.feature_names_, version=new_version)
                        else:
                            if hasattr(new_model, 'feature_importances_'):
                                generate_feature_importance_plot(new_model, features.columns, version=new_version)
                            
                        generate_prediction_vs_actual_plot(y_test, y_pred_test, version=new_version)
                        generate_error_distribution_plot(y_test, y_pred_test, version=new_version)
                        
                        # Show success message with metrics
                        st.success(f"Successfully trained and saved model version {new_version}")
                        st.info(f"Test set R² score: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.6f}")
                        st.info("Refresh the page to select the new model version")
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    logger.error(f"Error training model: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Model Parameter Information
        if model_loaded:
            with st.expander("Current Model Parameters", expanded=False):
                metadata = load_model_metadata(selected_version)
                if metadata and 'model_parameters' in metadata:
                    params = metadata['model_parameters']
                    st.json(params)
                    
                    if 'sample_weights' in metadata:
                        st.subheader("Sample Weight Configuration")
                        st.json(metadata['sample_weights'])
                else:
                    st.write("Model parameters not available in metadata.")
                    st.write("Current model parameters:")
                    st.write(model.get_params())
        
        # Cross-validation results
        st.subheader("Cross-Validation Performance (5-fold on Training Set)")
        col1, col2 = st.columns(2)
        col1.metric("Average Mean Squared Error", f"{cv_results['mse']['mean']:.6f}")
        col1.metric("Average Root MSE", f"{cv_results['rmse']['mean']:.6f}")
        col2.metric("Average Mean Absolute Error", f"{cv_results['mae']['mean']:.6f}")
        col2.metric("Average R² Score", f"{cv_results['r2']['mean']:.4f}")
        
        # Test set results
        st.subheader("Test Set Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{test_metrics['mse']:.6f}")
        col1.metric("Root MSE", f"{test_metrics['rmse']:.6f}")
        col2.metric("Mean Absolute Error", f"{test_metrics['mae']:.6f}")
        col2.metric("R² Score", f"{test_metrics['r2']:.4f}")
        
        # Full dataset verification
        st.subheader("Full Dataset Verification")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{full_metrics['mse']:.6f}")
        col1.metric("Root MSE", f"{full_metrics['rmse']:.6f}")
        col2.metric("Mean Absolute Error", f"{full_metrics['mae']:.6f}")
        col2.metric("R² Score", f"{full_metrics['r2']:.4f}")
        
        # Age group heatmap
        with st.expander("Age Group Analysis", expanded=False):
            try:
                st.write("This analysis shows engagement metrics (Open rate, Click rate, and Opt-out rate) by age group.")
                
                # Process data for heatmap
                heatmap_data, results_df = process_data_for_age_heatmap(delivery_data, customer_data)
                
                # Display selector for what to show in heatmap
                view_options = ["Overall"]
                view_options.extend([col for col in heatmap_data['Openrate'].columns if col != "Overall"])
                selected_views = st.multiselect("Select views to display", options=view_options, default=["Overall"])
                
                if selected_views:
                    # Filter data to selected views
                    filtered_data = {
                        metric: data[selected_views] for metric, data in heatmap_data.items()
                    }
                    
                    # Create tabs for different metrics
                    metric_tabs = st.tabs(["Open Rate", "Click Rate", "Opt-out Rate"])
                    
                    with metric_tabs[0]:
                        fig = create_interactive_heatmap(filtered_data['Openrate'], 'Openrate', 'Open Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_tabs[1]:
                        fig = create_interactive_heatmap(filtered_data['Clickrate'], 'Clickrate', 'Click Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_tabs[2]:
                        fig = create_interactive_heatmap(filtered_data['Optoutrate'], 'Optoutrate', 'Opt-out Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw data
                    if st.checkbox("Show raw data"):
                        st.dataframe(results_df)
                else:
                    st.info("Please select at least one view to display.")
            except Exception as e:
                st.error(f"Error in age group analysis: {str(e)}")
                logger.error(f"Error in age group analysis: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Feature importances
        st.subheader("Feature Importances")
        
        try:
            feature_importance_df = extract_feature_importance(model, features.columns)
            
            # Plot top features
            top_features = feature_importance_df.head(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_features['Feature'],
                x=top_features['Importance'],
                orientation='h',
                marker_color='#1f77b4'
            ))
            
            fig.update_layout(
                title='Top 15 Feature Importances',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show all feature importances in expandable section
            with st.expander("All Feature Importances", expanded=False):
                st.dataframe(feature_importance_df)
                
                # Download as CSV
                csv_data = feature_importance_df.to_csv(index=False)
                st.download_button(
                    label="Download Feature Importances",
                    data=csv_data,
                    file_name="feature_importances.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error displaying feature importances: {str(e)}")
        
        # Dataset information
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {features.shape[0]}")
        st.write(f"Number of features: {features.shape[1]}")
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual Values (Test Set)")
        try:
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred_test,
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.6,
                    line=dict(width=1),
                    color='#1f77b4'
                ),
                name='Predictions'
            ))
            
            # Add ideal line
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Ideal'
            ))
            
            # Update layout
            fig.update_layout(
                title='Actual vs Predicted Open Rates (Test Set)',
                xaxis_title='Actual Open Rate',
                yaxis_title='Predicted Open Rate',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying predictions vs actual plot: {str(e)}")
        
        # Error distribution
        st.subheader("Distribution of Prediction Errors (Test Set)")
        try:
            errors = y_test - y_pred_test
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                marker_color='#1f77b4',
                opacity=0.75
            ))
            
            # Add vertical line at zero
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="red"
            )
            
            # Update layout
            fig.update_layout(
                title='Distribution of Prediction Errors (Test Set)',
                xaxis_title='Prediction Error',
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error statistics
            error_stats = {
                'Mean Error': errors.mean(),
                'Std Dev': errors.std(),
                'Min Error': errors.min(),
                'Max Error': errors.max(),
                'Median Error': np.median(errors)
            }
            
            # Display as metrics
            cols = st.columns(len(error_stats))
            for i, (label, value) in enumerate(error_stats.items()):
                cols[i].metric(label, f"{value:.4f}")
            
        except Exception as e:
            st.error(f"Error displaying error distribution plot: {str(e)}")
        
        # Documentation links
        st.subheader("Model Documentation")
        doc_path = os.path.join('Docs', f"model_v{selected_version}")
        if os.path.exists(doc_path):
            st.info(f"Documentation available at {doc_path}")
            
            # List documentation files
            doc_files = os.listdir(doc_path)
            if doc_files:
                st.write("Available documentation files:")
                for file in doc_files:
                    st.write(f"- {file}")
                    
                # Add option to download documentation
                doc_file = os.path.join(doc_path, "model_documentation.yaml")
                if os.path.exists(doc_file):
                    with open(doc_file, 'r') as f:
                        doc_content = f.read()
                    st.download_button(
                        label="Download Model Documentation",
                        data=doc_content,
                        file_name=f"model_v{selected_version}_documentation.yaml",
                        mime="text/yaml"
                    )
                    
                # Display feature importance plot if available
                importance_plot = os.path.join(doc_path, "feature_importance_plot.png")
                pred_vs_actual_plot = os.path.join(doc_path, "prediction_vs_actual_plot.png")
                
                if os.path.exists(importance_plot) and os.path.exists(pred_vs_actual_plot):
                    st.subheader("Saved Visualization")
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(importance_plot, "rb") as file:
                            st.image(file.read(), caption="Saved Feature Importance")
                        
                    with col2:
                        with open(pred_vs_actual_plot, "rb") as file:
                            st.image(file.read(), caption="Saved Prediction vs Actual")
            else:
                st.write("No documentation files found.")
        else:
            st.info("No documentation available for this model version.")
            
    # Tab 4: KPI Dashboard
    with tab4:
        try:
            create_kpi_dashboard(delivery_data, threshold=0.5)
        except Exception as e:
            st.error(f"Error creating KPI dashboard: {str(e)}")
            logger.error(f"Error creating KPI dashboard: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Tab 2: Batch Processing
    with tab2:
        try:
            create_batch_prediction_ui(model, selected_version, include_preheader=include_preheader)
        except Exception as e:
            st.error(f"Error in batch prediction UI: {str(e)}")
            logger.error(f"Error in batch prediction UI: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Tab 1: Sendout Prediction
    with tab1:
        st.header('Predict KPIs for New Sendout')
        st.info(f"Using model version {selected_version}")
        
        # Get feature metadata from model or fallback to current feature metadata
        model_metadata = load_model_metadata(selected_version)
        if model_metadata and 'feature_names' in model_metadata:
            logger.info(f"Using feature names from model metadata: {len(model_metadata['feature_names'])} features")
        else:
            logger.info(f"Using current feature metadata")
        
        # UI for prediction inputs
        dialog_options = sorted(delivery_data['Dialog'].unique().tolist())
        dialog_labels = []
        
        for d in dialog_options:
            dialog_display = d
            for key, value in DIALOG_VALUES.items():
                if value[0] == d:
                    dialog_display = value[1]
                    break
            dialog_labels.append((d, dialog_display))
        
        selected_dialog_display = st.selectbox('Dialog', options=[label for _, label in dialog_labels])
        selected_dialog_code = next(code for code, label in dialog_labels if label == selected_dialog_display)
        
        syfte_options = sorted(delivery_data['Syfte'].unique().tolist())
        syfte_labels = []
        
        for s in syfte_options:
            syfte_display = s
            for key, value in SYFTE_VALUES.items():
                if value[0] == s:
                    syfte_display = value[1]
                    break
            syfte_labels.append((s, syfte_display))
        
        selected_syfte_display = st.selectbox('Syfte', options=[label for _, label in syfte_labels])
        selected_syfte_code = next(code for code, label in syfte_labels if label == selected_syfte_display)
        
        product_options = sorted(delivery_data['Product'].unique().tolist())
        product_labels = []
        
        for p in product_options:
            product_display = p
            for key, value in PRODUKT_VALUES.items():
                if value[0] == p:
                    product_display = value[1]
                    break
            product_labels.append((p, product_display))
        
        selected_product_display = st.selectbox('Product', options=[label for _, label in product_labels])
        selected_product_code = next(code for code, label in product_labels if label == selected_product_display)
        
        bolag_options = sorted(customer_data['Bolag'].unique().tolist())
        excluded_bolag_display = st.multiselect('Exclude Bolag', bolag_options)
        included_bolag = [b for b in bolag_options if b not in excluded_bolag_display]
        
        min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
        max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
        
        # Enhanced subject line and preheader input with GenAI checkbox
        subject_line, preheader, use_genai, genai_options = create_enhanced_input_ui()
        
        # Prediction logic
        if subject_line and (preheader or float(selected_version.split('.')[0]) < 2):
            try:
                # Create base input data based on model version
                if float(selected_version.split('.')[0]) >= 2:
                    base_input_data = pd.DataFrame(columns=features_dict['v2'].columns)
                else:
                    base_input_data = pd.DataFrame(columns=features_dict['legacy'].columns)
                    
                base_input_data.loc[0] = 0
                
                # Get dummy column names
                if hasattr(model, 'feature_names_'):
                    model_columns = model.feature_names_
                    
                    dialog_col = f'Dialog_{selected_dialog_code}'
                    syfte_col = f'Syfte_{selected_syfte_code}'
                    product_col = f'Product_{selected_product_code}'
                    
                    # Check if columns exist in model features
                    dialog_exists = dialog_col in model_columns
                    syfte_exists = syfte_col in model_columns
                    product_exists = product_col in model_columns
                    
                    # Only set columns that exist in the model
                    if dialog_exists and dialog_col in base_input_data.columns:
                        base_input_data[dialog_col] = 1
                    if syfte_exists and syfte_col in base_input_data.columns:
                        base_input_data[syfte_col] = 1
                    if product_exists and product_col in base_input_data.columns:
                        base_input_data[product_col] = 1
                else:
                    dialog_col = f'Dialog_{selected_dialog_code}'
                    syfte_col = f'Syfte_{selected_syfte_code}'
                    product_col = f'Product_{selected_product_code}'
                    
                    if dialog_col in base_input_data.columns:
                        base_input_data[dialog_col] = 1
                    else:
                        st.warning(f"Column '{dialog_col}' not found. Using available dialog columns.")
                        dialog_cols = [col for col in base_input_data.columns if col.startswith('Dialog_')]
                        if dialog_cols and st.checkbox(f"Use first available dialog column: {dialog_cols[0]}", value=True):
                            base_input_data[dialog_cols[0]] = 1
                    
                    if syfte_col in base_input_data.columns:
                        base_input_data[syfte_col] = 1
                    else:
                        st.warning(f"Column '{syfte_col}' not found. Using available syfte columns.")
                        syfte_cols = [col for col in base_input_data.columns if col.startswith('Syfte_')]
                        if syfte_cols and st.checkbox(f"Use first available syfte column: {syfte_cols[0]}", value=True):
                            base_input_data[syfte_cols[0]] = 1
                    
                    if product_col in base_input_data.columns:
                        base_input_data[product_col] = 1
                    else:
                        st.warning(f"Column '{product_col}' not found. Using available product columns.")
                        product_cols = [col for col in base_input_data.columns if col.startswith('Product_')]
                        if product_cols and st.checkbox(f"Use first available product column: {product_cols[0]}", value=True):
                            base_input_data[product_cols[0]] = 1
                
                # Set age features
                if 'Min_age' in base_input_data.columns:
                    base_input_data['Min_age'] = min_age
                if 'Max_age' in base_input_data.columns:
                    base_input_data['Max_age'] = max_age
                
                # Set bolag features
                for b in included_bolag:
                    bolag_col = f'Bolag_{b}'
                    if bolag_col in base_input_data.columns:
                        base_input_data[bolag_col] = 1
                
                # Prediction function based on model version
                if float(selected_version.split('.')[0]) >= 2:
                    def predict_for_subject_and_preheader(subject_line, preheader):
                        input_data = base_input_data.copy()
                        
                        # Set subject features
                        input_data['Subject_length'] = len(subject_line)
                        input_data['Subject_num_words'] = len(subject_line.split())
                        input_data['Subject_has_exclamation'] = 1 if '!' in subject_line else 0
                        input_data['Subject_has_question'] = 1 if '?' in subject_line else 0
                        
                        # Set advanced subject features
                        input_data['Subject_caps_ratio'] = sum(1 for c in str(subject_line) if c.isupper()) / len(str(subject_line)) if len(str(subject_line)) > 0 else 0
                        input_data['Subject_avg_word_len'] = np.mean([len(w) for w in str(subject_line).split()]) if len(str(subject_line).split()) > 0 else 0
                        input_data['Subject_num_special_chars'] = sum(1 for c in str(subject_line) if c in '!?%$€£#@*&')
                        input_data['Subject_first_word_len'] = len(str(subject_line).split()[0]) if len(str(subject_line).split()) > 0 else 0
                        input_data['Subject_last_word_len'] = len(str(subject_line).split()[-1]) if len(str(subject_line).split()) > 0 else 0
                        
                        # Set preheader features
                        input_data['Preheader_length'] = len(preheader)
                        input_data['Preheader_num_words'] = len(preheader.split())
                        input_data['Preheader_has_exclamation'] = 1 if '!' in preheader else 0
                        input_data['Preheader_has_question'] = 1 if '?' in preheader else 0
                        
                        # Set advanced preheader features
                        input_data['Preheader_caps_ratio'] = sum(1 for c in str(preheader) if c.isupper()) / len(str(preheader)) if len(str(preheader)) > 0 else 0
                        input_data['Preheader_avg_word_len'] = np.mean([len(w) for w in str(preheader).split()]) if len(str(preheader).split()) > 0 else 0
                        input_data['Preheader_num_special_chars'] = sum(1 for c in str(preheader) if c in '!?%$€£#@*&')
                        
                        # Set relationship features
                        preheader_len = len(preheader) if len(preheader) > 0 else 1
                        preheader_words = len(preheader.split()) if len(preheader.split()) > 0 else 1
                        input_data['Subject_preheader_length_ratio'] = len(subject_line) / preheader_len
                        input_data['Subject_preheader_words_ratio'] = len(subject_line.split()) / preheader_words
                        
                        # Check if the input data matches the model's expected columns
                        if hasattr(model, 'feature_names_'):
                            adapted_input = adapt_features_to_model(model, input_data)
                            return model.predict(adapted_input)[0]
                        else:
                            return model.predict(input_data)[0]
                else:
                    def predict_for_subject_and_preheader(subject_line, preheader=None):
                        input_data = base_input_data.copy()
                        
                        # Use legacy feature names
                        input_data['Subject_length'] = len(subject_line)
                        input_data['Num_words'] = len(subject_line.split())
                        input_data['Has_exclamation'] = 1 if '!' in subject_line else 0
                        input_data['Has_question'] = 1 if '?' in subject_line else 0
                        
                        # Check if the input data matches the model's expected columns
                        if hasattr(model, 'feature_names_'):
                            adapted_input = adapt_features_to_model(model, input_data)
                            return model.predict(adapted_input)[0]
                        else:
                            return model.predict(input_data)[0]
                
                # Calculate KPIs for current subject line and preheader
                openrate_A = predict_for_subject_and_preheader(subject_line, preheader)
                avg_clickrate = delivery_data['Clickrate'].mean()
                avg_optoutrate = delivery_data['Optoutrate'].mean()
                
                # Display predicted results for current subject line and preheader
                st.subheader('Predicted Results')
                col1, col2, col3 = st.columns(3)
                col1.metric("Open Rate", f"{openrate_A:.2%}")
                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                
                # Age group analysis
                try:
                    # Process data for heatmap
                    heatmap_data, _ = process_data_for_age_heatmap(delivery_data, customer_data)
                    
                    with st.expander("Age Group Analysis", expanded=False):
                        st.write("This analysis shows open rate, click rate, and opt-out rate by age group.")
                        
                        # Create tabs for different metrics
                        metric_tabs = st.tabs(["Open Rate", "Click Rate", "Opt-out Rate"])
                        
                        with metric_tabs[0]:
                            open_data = heatmap_data['Openrate'][['Overall']]
                            fig = create_interactive_heatmap(open_data, 'Openrate', 'Open Rate by Age Group')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with metric_tabs[1]:
                            click_data = heatmap_data['Clickrate'][['Overall']]
                            fig = create_interactive_heatmap(click_data, 'Clickrate', 'Click Rate by Age Group')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with metric_tabs[2]:
                            optout_data = heatmap_data['Optoutrate'][['Overall']]
                            fig = create_interactive_heatmap(optout_data, 'Optoutrate', 'Opt-out Rate by Age Group')
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in age group analysis: {str(e)}")
                    logger.error(f"Error in age group analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # A/B/C/D Testing with Groq API including Preheader
                if use_genai:
                    if st.button('Send to Groq API'):
                        with st.spinner("Generating alternatives..."):
                            response_data = send_to_groq_api(
                                subject_line, preheader, 
                                openrate_A, 
                                selected_dialog_display, selected_syfte_display, selected_product_display,
                                min_age, max_age, 
                                included_bolag,
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
                                    
                                    options = []
                                    for i, sug in enumerate(suggestions[:3], start=1):
                                        subject = sug.get('subject', '')
                                        preheader_alt = sug.get('preheader', '')
                                        
                                        if subject and (preheader_alt or float(selected_version.split('.')[0]) < 2):
                                            openrate = predict_for_subject_and_preheader(subject, preheader_alt)
                                            options.append((chr(65 + i), subject, preheader_alt, openrate))
                                    
                                    if options:
                                        # Add current option as Version A
                                        all_options = [('A', subject_line, preheader, openrate_A)] + options
                                        
                                        # Use enhanced visualization
                                        display_abcd_results(all_options, avg_clickrate, avg_optoutrate, heatmap_data)
                                        
                                        # Version comparison heatmaps
                                        st.subheader("Age Group Analysis Across Versions")
                                        try:
                                            # Create tabs for different metrics
                                            metric_tabs = st.tabs(["Open Rate", "Click Rate", "Opt-out Rate"])
                                            
                                            with metric_tabs[0]:
                                                # Prepare data with all versions using proportional estimates
                                                version_open_data = prepare_version_heatmap_data(all_options, heatmap_data, 'Openrate')
                                                
                                                # Calculate the version with best performance for each age group
                                                best_version_by_age = version_open_data.idxmax(axis=1)
                                                
                                                fig = create_interactive_heatmap(version_open_data, 'Openrate', 
                                                                            'Open Rate by Age Group and Version',
                                                                            colorscale='Viridis')
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Show which version is best for each age group
                                                st.subheader("Best Version by Age Group (Open Rate)")
                                                best_df = pd.DataFrame({
                                                    'Age Group': best_version_by_age.index,
                                                    'Best Version': best_version_by_age.values,
                                                    'Estimated Open Rate': [version_open_data.loc[age, ver] for age, ver in zip(best_version_by_age.index, best_version_by_age.values)]
                                                })
                                                st.dataframe(best_df.set_index('Age Group'), use_container_width=True)
                                            
                                            with metric_tabs[1]:
                                                version_click_data = prepare_version_heatmap_data(all_options, heatmap_data, 'Clickrate')
                                                fig = create_interactive_heatmap(version_click_data, 'Clickrate', 
                                                                            'Click Rate by Age Group and Version',
                                                                            colorscale='Blues')
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            with metric_tabs[2]:
                                                version_optout_data = prepare_version_heatmap_data(all_options, heatmap_data, 'Optoutrate')
                                                fig = create_interactive_heatmap(version_optout_data, 'Optoutrate', 
                                                                            'Opt-out Rate by Age Group and Version',
                                                                            colorscale='Reds')
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                            st.caption("""Note: These heatmaps show estimated performance by age group for each version.
                                        The estimates are based on the overall predicted open rate and how it might affect different age groups proportionally.
                                        For open rates, the estimations apply the ratio between predicted rates to the baseline age distribution.
                                        For click and opt-out rates, simulated effects are derived from the open rate changes.""")
                                        except Exception as e:
                                            st.error(f"Error displaying age group heatmaps: {str(e)}")
                                            logger.error(f"Error displaying age group heatmaps: {str(e)}")
                                            logger.error(traceback.format_exc())
                                except Exception as e:
                                    st.error(f"Error processing alternatives: {str(e)}")
                                    logger.error(f"Error processing alternatives: {str(e)}")
                                    logger.error(traceback.format_exc())

            except Exception as e:
                st.error(f"Error setting up prediction: {str(e)}")
                logger.error(f"Error setting up prediction: {str(e)}")
                logger.error(traceback.format_exc())
                return

if __name__ == '__main__':
    main()