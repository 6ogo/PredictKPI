"""
Model training, evaluation, and management for KPI Predictor application
"""
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import yaml
import logging
import traceback
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from utils import logger, get_model_filename, get_model_metadata_filename, save_model_metadata

# --- Model Factory ---
class ModelFactory:
    """Factory class to create different types of models with consistent interface"""
    
    @staticmethod
    def create_model(model_type='xgboost', params=None):
        """
        Create a regression model of the specified type
        
        Parameters:
        model_type (str): Type of model ('xgboost', 'lightgbm', 'catboost', 'ensemble')
        params (dict): Parameters for the model
        
        Returns:
        object: Model instance with fit and predict methods
        """
        if params is None:
            params = {}
            
        default_params = {
            'random_state': 42
        }
        
        # Merge default params with provided params
        merged_params = {**default_params, **params}
        
        if model_type == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(**merged_params)
            
        elif model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(**merged_params)
            except ImportError:
                logger.warning("LightGBM not installed. Falling back to XGBoost.")
                from xgboost import XGBRegressor
                return XGBRegressor(**merged_params)
                
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                # CatBoost has specific defaults that work well
                cb_params = {**merged_params}
                # Remove params that cause issues with CatBoost if present
                for param in ['reg_lambda', 'subsample', 'colsample_bytree']:
                    if param in cb_params:
                        del cb_params[param]
                return CatBoostRegressor(**cb_params, verbose=False)
            except ImportError:
                logger.warning("CatBoost not installed. Falling back to XGBoost.")
                from xgboost import XGBRegressor
                return XGBRegressor(**merged_params)
                
        elif model_type == 'ensemble':
            # Create a simple ensemble of models
            try:
                from sklearn.ensemble import StackingRegressor
                from xgboost import XGBRegressor
                
                estimators = []
                
                # Try to add LightGBM
                try:
                    from lightgbm import LGBMRegressor
                    estimators.append(('lgbm', LGBMRegressor(**merged_params)))
                except ImportError:
                    pass
                    
                # Try to add CatBoost
                try:
                    from catboost import CatBoostRegressor
                    cb_params = {**merged_params}
                    for param in ['reg_lambda', 'subsample', 'colsample_bytree']:
                        if param in cb_params:
                            del cb_params[param]
                    estimators.append(('catboost', CatBoostRegressor(**cb_params, verbose=False)))
                except ImportError:
                    pass
                
                # Always add XGBoost
                estimators.append(('xgb', XGBRegressor(**merged_params)))
                
                # Add a simple model as regularization
                from sklearn.linear_model import RidgeCV
                estimators.append(('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0])))
                
                # Create the ensemble
                return StackingRegressor(
                    estimators=estimators,
                    final_estimator=XGBRegressor(**merged_params)
                )
            except Exception as e:
                logger.warning(f"Error creating ensemble model: {str(e)}. Falling back to XGBoost.")
                from xgboost import XGBRegressor
                return XGBRegressor(**merged_params)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using XGBoost.")
            from xgboost import XGBRegressor
            return XGBRegressor(**merged_params)

# --- Neural Network Model ---
def create_neural_network_model(input_dim, params=None):
    """Create a neural network regression model using TensorFlow/Keras if available"""
    if params is None:
        params = {}
    
    learning_rate = params.get('learning_rate', 0.001)
    hidden_layers = params.get('hidden_layers', [64, 32])
    dropout_rate = params.get('dropout_rate', 0.2)
    activation = params.get('activation', 'relu')
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
        
        def create_model():
            model = Sequential()
            
            # First layer
            model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            # Hidden layers
            for units in hidden_layers[1:]:
                model.add(Dense(units, activation=activation))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            # Output layer
            model.add(Dense(1, activation='linear'))
            
            # Compile model
            model.compile(
                loss='mean_squared_error',
                optimizer=Adam(learning_rate=learning_rate)
            )
            
            return model
        
        # Create a scikit-learn compatible model
        nn_model = KerasRegressor(
            build_fn=create_model,
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=params.get('patience', 10),
                    restore_best_weights=True
                )
            ]
        )
        
        return nn_model
    
    except ImportError:
        logger.warning("TensorFlow not installed. Falling back to XGBoost.")
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42)

# --- Model Training ---
def train_model(X_train, y_train, sample_weights=None, params=None, model_type='xgboost'):
    """
    Train a model with the given parameters
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
    sample_weights : array-like, optional
        Sample weights for training
    params : dict, optional
        Model parameters
    model_type : str, optional
        Type of model to train ('xgboost', 'lightgbm', 'catboost', 'ensemble', 'nn')
        
    Returns:
    --------
    object
        Trained model
    """
    try:
        if params is None:
            params = {
                'reg_lambda': 1.0,
                'random_state': 42
            }
        
        if model_type == 'nn':
            # Neural network model
            input_dim = X_train.shape[1]
            model = create_neural_network_model(input_dim, params)
        else:
            # Tree-based model
            model = ModelFactory.create_model(model_type, params)
        
        logger.info(f"Training {model_type} model with parameters: {params}")
        
        if model_type == 'nn':
            # For neural network, use validation split
            validation_size = min(0.2, 1000 / len(X_train)) if len(X_train) > 1000 else 0.2
            X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                X_train, y_train, test_size=validation_size, random_state=42
            )
            model.fit(
                X_train_nn, y_train_nn,
                validation_data=(X_val_nn, y_val_nn),
                epochs=params.get('epochs', 100),
                batch_size=params.get('batch_size', 32),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=params.get('patience', 10),
                        restore_best_weights=True
                    )
                ]
            )
        else:
            # For tree-based models
            model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model_with_params(X_train, y_train, params, sample_weight_config=None, model_type='xgboost'):
    """
    Train a model with the given parameters and sample weight configuration
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
    params : dict
        Model parameters
    sample_weight_config : dict, optional
        Configuration for sample weights including threshold and weight ratio
    model_type : str, optional
        Type of model to train
        
    Returns:
    --------
    object
        Trained model
    """
    try:
        # Configure sample weights if provided
        sample_weights = None
        if sample_weight_config is not None:
            threshold = sample_weight_config.get('threshold', 0.5)
            weight_high = sample_weight_config.get('weight_high', 2.0)
            weight_low = sample_weight_config.get('weight_low', 1.0)
            
            logger.info(f"Using sample weights - threshold: {threshold}, high: {weight_high}, low: {weight_low}")
            sample_weights = np.where(y_train > threshold, weight_high, weight_low)
        
        # Train model
        model = train_model(X_train, y_train, sample_weights, params, model_type)
        
        return model
    except Exception as e:
        logger.error(f"Error training model with parameters: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Model Feature Utilities ---
def extract_feature_importance(model, feature_names):
    """Extract feature importance from different model types in a consistent format"""
    if hasattr(model, 'feature_importances_'):
        # Standard sklearn-like API
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    elif hasattr(model, 'named_estimators_'):
        # It's a stacking or ensemble model
        # Get feature importance from the final estimator if possible
        if hasattr(model.final_estimator_, 'feature_importances_'):
            # For StackingRegressor, this is not directly linked to original features
            return pd.DataFrame({
                'Feature': [f'Meta_{i}' for i in range(len(model.final_estimator_.feature_importances_))],
                'Importance': model.final_estimator_.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            # Try to get importance from the first base estimator that has it
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    return pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': estimator.feature_importances_
                    }).sort_values('Importance', ascending=False)
    
    # Fallback - return placeholder importance
    logger.warning("Could not extract feature importance from model")
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.ones(len(feature_names)) / len(feature_names)
    })

def validate_model_features(model, features):
    """
    Validate that the features match what the model expects
    """
    try:
        if not hasattr(model, 'feature_names_'):
            logger.warning("Model doesn't have feature_names_ attribute. Skipping validation.")
            return True, "Feature validation skipped"
        
        model_features = model.feature_names_
        input_features = features.columns.tolist()
        
        if set(model_features) != set(input_features):
            missing_in_input = set(model_features) - set(input_features)
            extra_in_input = set(input_features) - set(model_features)
            
            error_msg = "Feature mismatch between model and input data.\n"
            if missing_in_input:
                error_msg += f"Features missing in input: {missing_in_input}\n"
            if extra_in_input:
                error_msg += f"Extra features in input: {extra_in_input}\n"
                
            logger.error(error_msg)
            return False, error_msg
        
        return True, "Features validated successfully"
    except Exception as e:
        logger.error(f"Error validating model features: {str(e)}")
        logger.error(traceback.format_exc())
        return False, str(e)

def adapt_features_to_model(model, features):
    """
    Adapt the input features to match what the model expects
    """
    try:
        if not hasattr(model, 'feature_names_'):
            logger.warning("Model doesn't have feature_names_ attribute. Cannot adapt features.")
            return features
        
        model_features = model.feature_names_
        
        # Create a new DataFrame with the expected columns
        adapted_features = pd.DataFrame(index=features.index)
        
        for feature in model_features:
            if feature in features.columns:
                adapted_features[feature] = features[feature]
            else:
                # If the feature is missing, fill with zeros
                logger.warning(f"Feature '{feature}' missing in input data. Filling with zeros.")
                adapted_features[feature] = 0
        
        return adapted_features
    except Exception as e:
        logger.error(f"Error adapting features to model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    
    Returns:
    --------
    dict
        Dictionary of metrics
    array
        Predicted values
    """
    try:
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics, y_pred
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cross_validate_model(X_train, y_train, params=None, n_splits=5, sample_weights=None):
    """
    Perform cross-validation on the training data
    
    Returns:
    --------
    dict
        Dictionary of cross-validation results
    """
    try:
        if params is None:
            params = {
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_mse_scores = []
        cv_rmse_scores = []
        cv_mae_scores = []
        cv_r2_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # If sample weights are provided, subset them for this fold
            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights[train_idx]
            
            model_cv = XGBRegressor(**params)
            model_cv.fit(X_train_fold, y_train_fold, sample_weight=fold_weights)
            
            y_pred_val = model_cv.predict(X_val_fold)
            
            mse_cv = mean_squared_error(y_val_fold, y_pred_val)
            rmse_cv = np.sqrt(mse_cv)
            mae_cv = mean_absolute_error(y_val_fold, y_pred_val)
            r2_cv = r2_score(y_val_fold, y_pred_val)
            
            cv_mse_scores.append(mse_cv)
            cv_rmse_scores.append(rmse_cv)
            cv_mae_scores.append(mae_cv)
            cv_r2_scores.append(r2_cv)
        
        cv_results = {
            'mse': {
                'scores': cv_mse_scores,
                'mean': np.mean(cv_mse_scores),
                'std': np.std(cv_mse_scores)
            },
            'rmse': {
                'scores': cv_rmse_scores,
                'mean': np.mean(cv_rmse_scores),
                'std': np.std(cv_rmse_scores)
            },
            'mae': {
                'scores': cv_mae_scores,
                'mean': np.mean(cv_mae_scores),
                'std': np.std(cv_mae_scores)
            },
            'r2': {
                'scores': cv_r2_scores,
                'mean': np.mean(cv_r2_scores),
                'std': np.std(cv_r2_scores)
            }
        }
        
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cross_validate_model_parallel(X_train, y_train, params=None, n_splits=5, sample_weights=None, n_jobs=-1):
    """
    Perform cross-validation on the training data using parallel processing
    """
    try:
        from joblib import Parallel, delayed
        
        if params is None:
            params = {
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X_train))
        
        def evaluate_fold(train_idx, val_idx):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # If sample weights are provided, subset them for this fold
            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights[train_idx]
            
            model_cv = XGBRegressor(**params)
            model_cv.fit(X_train_fold, y_train_fold, sample_weight=fold_weights)
            
            y_pred_val = model_cv.predict(X_val_fold)
            
            mse_cv = mean_squared_error(y_val_fold, y_pred_val)
            rmse_cv = np.sqrt(mse_cv)
            mae_cv = mean_absolute_error(y_val_fold, y_pred_val)
            r2_cv = r2_score(y_val_fold, y_pred_val)
            
            return {
                'mse': mse_cv,
                'rmse': rmse_cv,
                'mae': mae_cv,
                'r2': r2_cv
            }
        
        # Run folds in parallel
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(train_idx, val_idx) for train_idx, val_idx in splits
        )
        
        # Collect results
        cv_results = {
            'mse': {
                'scores': [r['mse'] for r in fold_results],
                'mean': np.mean([r['mse'] for r in fold_results]),
                'std': np.std([r['mse'] for r in fold_results])
            },
            'rmse': {
                'scores': [r['rmse'] for r in fold_results],
                'mean': np.mean([r['rmse'] for r in fold_results]),
                'std': np.std([r['rmse'] for r in fold_results])
            },
            'mae': {
                'scores': [r['mae'] for r in fold_results],
                'mean': np.mean([r['mae'] for r in fold_results]),
                'std': np.std([r['mae'] for r in fold_results])
            },
            'r2': {
                'scores': [r['r2'] for r in fold_results],
                'mean': np.mean([r['r2'] for r in fold_results]),
                'std': np.std([r['r2'] for r in fold_results])
            }
        }
        
        return cv_results
    except Exception as e:
        logger.error(f"Error in parallel cross-validation: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to non-parallel version
        return cross_validate_model(X_train, y_train, params, n_splits, sample_weights)

def tune_hyperparameters(X_train, y_train, param_grid=None, cv=5, sample_weights=None):
    """
    Tune model hyperparameters using GridSearchCV
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
    param_grid : dict
        Parameter grid for GridSearchCV
    cv : int
        Number of cross-validation folds
    sample_weights : array-like
        Sample weights for training
        
    Returns:
    --------
    dict
        Best parameters found by GridSearchCV
    float
        Best score
    """
    try:
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.04, 0.1],
                'reg_lambda': [0.5, 1.0, 1.7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
        logger.info(f"Starting hyperparameter tuning with param grid: {param_grid}")
        
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Implement GridSearchCV with early stopping
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit with sample weights if provided
        if sample_weights is not None:
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best RMSE score: {-best_score:.6f}")
        
        return best_params, best_score
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Documentation ---
def generate_model_documentation(model, feature_metadata, train_metrics, cv_results, test_metrics, version=None):
    """
    Generate model documentation
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_metadata : dict
        Feature metadata
    train_metrics : dict
        Training metrics
    cv_results : dict
        Cross-validation results
    test_metrics : dict
        Test metrics
    version : str, optional
        Model version
        
    Returns:
    --------
    str
        Path to documentation file
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create documentation directory
        model_doc_dir = os.path.join('Docs', f"model_v{version}")
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # General model information
        model_info = {
            'version': version,
            'created_at': timestamp,
            'xgboost_version': model.__class__.__module__,
            'model_parameters': model.get_params(),
            'feature_count': len(model.feature_names_) if hasattr(model, 'feature_names_') else "Unknown",
        }
        
        # Feature information
        feature_info = {
            'feature_names': model.feature_names_ if hasattr(model, 'feature_names_') else "Unknown",
            'metadata': feature_metadata
        }
        
        # Performance metrics
        performance_metrics = {
            'training': train_metrics,
            'cross_validation': cv_results,
            'test': test_metrics
        }
        
        # Save documentation
        documentation = {
            'model_info': model_info,
            'feature_info': feature_info,
            'performance_metrics': performance_metrics
        }
        
        doc_file = os.path.join(model_doc_dir, 'model_documentation.yaml')
        with open(doc_file, 'w') as f:
            yaml.dump(documentation, f)
        
        logger.info(f"Model documentation saved to {doc_file}")
        return doc_file
    except Exception as e:
        logger.error(f"Error generating model documentation: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_feature_importance_plot(model, feature_names, version=None):
    """
    Generate and save feature importance plot
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    version : str, optional
        Model version
        
    Returns:
    --------
    str
        Path to plot file
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join('Docs', f"model_v{version}")
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Save as CSV
        csv_file = os.path.join(model_doc_dir, 'feature_importances.csv')
        importance_df.to_csv(csv_file, index=False)
        
        # Generate plot for top 15 features
        top_features = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_features['Feature'], top_features['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_doc_dir, 'feature_importance_plot.png')
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        logger.info(f"Feature importance plot saved to {plot_file}")
        return plot_file
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_prediction_vs_actual_plot(y_test, y_pred, version=None):
    """
    Generate and save prediction vs actual plot
    
    Parameters:
    -----------
    y_test : Series
        Actual values
    y_pred : array
        Predicted values
    version : str, optional
        Model version
        
    Returns:
    --------
    str
        Path to plot file
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join('Docs', f"model_v{version}")
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel('Actual Open Rate')
        ax.set_ylabel('Predicted Open Rate')
        ax.set_title('Actual vs Predicted Open Rates (Test Set)')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_doc_dir, 'prediction_vs_actual_plot.png')
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        logger.info(f"Prediction vs actual plot saved to {plot_file}")
        return plot_file
    except Exception as e:
        logger.error(f"Error generating prediction vs actual plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_error_distribution_plot(y_test, y_pred, version=None):
    """
    Generate and save error distribution plot
    
    Parameters:
    -----------
    y_test : Series
        Actual values
    y_pred : array
        Predicted values
    version : str, optional
        Model version
        
    Returns:
    --------
    str
        Path to plot file
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join('Docs', f"model_v{version}")
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Calculate errors
        errors = y_test - y_pred
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors (Test Set)')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_doc_dir, 'error_distribution_plot.png')
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        logger.info(f"Error distribution plot saved to {plot_file}")
        return plot_file
    except Exception as e:
        logger.error(f"Error generating error distribution plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None