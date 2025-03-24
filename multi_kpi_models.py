"""
Enhanced models for multi-KPI prediction
"""
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, log_loss, accuracy_score
from sklearn.base import BaseEstimator

from utils import logger, get_model_directory, save_model_metadata, load_model_metadata


class MultiKpiModelManager:
    """
    Manager class for training, evaluating, and selecting the best models for multiple KPIs
    """
    
    def __init__(self, kpi_types=None):
        """
        Initialize with default KPI types if not specified
        """
        self.kpi_types = kpi_types or ['openrate', 'clickrate', 'optoutrate']
        self.models = {}
        self.model_metadata = {}
        self.best_model_info = {}
    
    def train_all_kpi_models(self, 
                            features_dict: Dict[str, pd.DataFrame], 
                            target_dict: Dict[str, pd.Series],
                            model_types: List[str] = None,
                            force_retrain: bool = False,
                            sample_weight_config: Dict[str, Any] = None,
                            version: str = None) -> Dict[str, BaseEstimator]:
        """
        Train models for all KPI types, selecting the best algorithm for each
        
        Parameters:
        -----------
        features_dict : Dict[str, pd.DataFrame]
            Dictionary mapping KPI type to feature DataFrames
        target_dict : Dict[str, pd.Series]
            Dictionary mapping KPI type to target Series
        model_types : List[str], optional
            List of model types to try ['xgboost', 'lightgbm', 'catboost', 'nn']
        force_retrain : bool
            Whether to force retraining even if models exist
        sample_weight_config : Dict[str, Any], optional
            Configuration for sample weights
        version : str, optional
            Model version string
            
        Returns:
        --------
        Dict[str, BaseEstimator]
            Dictionary of best trained models for each KPI
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'catboost']
            
        models = {}
        self.best_model_info = {}
        
        for kpi_type in self.kpi_types:
            if kpi_type not in features_dict or kpi_type not in target_dict:
                logger.warning(f"Missing features or target for {kpi_type}, skipping")
                continue
                
            logger.info(f"Training models for {kpi_type}...")
            
            # Load existing model if available and not forcing retrain
            model_file = self._get_model_path(kpi_type, version)
            if os.path.exists(model_file) and not force_retrain:
                logger.info(f"Loading existing model for {kpi_type} from {model_file}")
                try:
                    models[kpi_type] = joblib.load(model_file)
                    metadata = load_model_metadata(kpi_type, version)
                    if metadata:
                        self.model_metadata[kpi_type] = metadata
                        self.best_model_info[kpi_type] = {
                            'model_type': metadata.get('model_type', 'unknown'),
                            'performance': metadata.get('performance', {}).get('test', {})
                        }
                    continue
                except Exception as e:
                    logger.error(f"Error loading existing model: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue to train a new model
            
            X = features_dict[kpi_type]
            y = target_dict[kpi_type]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Try different model types
            best_model = None
            best_model_type = None
            best_score = float('-inf')  # For metrics like R2, AUC where higher is better
            best_performance = {}
            
            for model_type in model_types:
                logger.info(f"Trying {model_type} for {kpi_type}...")
                try:
                    # Configure model parameters based on KPI type
                    params = self._get_model_params(kpi_type, model_type)
                    
                    # Configure sample weights if needed
                    sample_weights = None
                    if sample_weight_config and kpi_type in sample_weight_config:
                        config = sample_weight_config[kpi_type]
                        threshold = config.get('threshold', 0.5)
                        weight_high = config.get('weight_high', 2.0)
                        weight_low = config.get('weight_low', 1.0)
                        sample_weights = np.where(y_train > threshold, weight_high, weight_low)
                    
                    # Create and train model
                    from models import ModelFactory
                    model = ModelFactory.create_model(model_type, params)
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                    
                    # Evaluate model
                    score, metrics = self._evaluate_model(model, X_test, y_test, kpi_type)
                    
                    logger.info(f"{model_type} for {kpi_type} - score: {score:.4f}, metrics: {metrics}")
                    
                    # Check if this is the best model so far
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_type = model_type
                        best_performance = metrics
                        
                except Exception as e:
                    logger.error(f"Error training {model_type} for {kpi_type}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            if best_model is not None:
                logger.info(f"Best model for {kpi_type}: {best_model_type} with score {best_score:.4f}")
                models[kpi_type] = best_model
                
                # Create metadata
                metadata = {
                    'kpi_type': kpi_type,
                    'model_type': best_model_type,
                    'version': version,
                    'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'feature_count': X.shape[1],
                    'training_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'feature_names': X.columns.tolist(),
                    'performance': {
                        'train': {},  # Could add train metrics later
                        'test': best_performance
                    },
                    'model_parameters': params
                }
                
                # Save model and metadata
                self._save_model(best_model, kpi_type, metadata, version)
                self.model_metadata[kpi_type] = metadata
                self.best_model_info[kpi_type] = {
                    'model_type': best_model_type,
                    'performance': best_performance
                }
            else:
                logger.error(f"Failed to train any model for {kpi_type}")
        
        self.models = models
        return models
    
    def predict_all_kpis(self, features_dict: Dict[str, pd.DataFrame], 
                        version: str = None) -> Dict[str, np.ndarray]:
        """
        Make predictions for all KPIs using trained models
        
        Parameters:
        -----------
        features_dict : Dict[str, pd.DataFrame]
            Dictionary mapping KPI type to feature DataFrames
        version : str, optional
            Model version string
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of predictions for each KPI
        """
        predictions = {}
        
        for kpi_type in self.kpi_types:
            if kpi_type not in features_dict:
                logger.warning(f"Missing features for {kpi_type}, skipping prediction")
                continue
                
            # Get model for this KPI
            model = self._get_model(kpi_type, version)
            
            if model is None:
                logger.warning(f"No model available for {kpi_type}, skipping prediction")
                continue
                
            # Make prediction
            try:
                X = features_dict[kpi_type]
                
                # Adapt features to match model if needed
                if hasattr(model, 'feature_names_'):
                    from models import adapt_features_to_model
                    X = adapt_features_to_model(model, X)
                
                # Make prediction
                predictions[kpi_type] = model.predict(X)
                
            except Exception as e:
                logger.error(f"Error predicting {kpi_type}: {str(e)}")
                logger.error(traceback.format_exc())
                predictions[kpi_type] = np.zeros(len(features_dict[kpi_type]))
        
        return predictions
    
    def predict_new_campaign(self, subject: str, preheader: str, 
                           campaign_metadata: Dict[str, Any], 
                           version: str = None) -> Dict[str, float]:
        """
        Predict KPIs for a new campaign
        
        Parameters:
        -----------
        subject : str
            Email subject line
        preheader : str
            Email preheader text
        campaign_metadata : Dict[str, Any]
            Dictionary of campaign metadata (dialog, syfte, etc.)
        version : str, optional
            Model version string
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of predicted KPIs
        """
        from features import create_campaign_level_features
        
        # Create features for new campaign
        features_dict = create_campaign_level_features(
            subject, preheader, campaign_metadata
        )
        
        # Make predictions
        predictions = {}
        
        for kpi_type in self.kpi_types:
            if kpi_type not in features_dict:
                logger.warning(f"Missing features for {kpi_type}, skipping prediction")
                continue
                
            # Get model for this KPI
            model = self._get_model(kpi_type, version)
            
            if model is None:
                logger.warning(f"No model available for {kpi_type}, skipping prediction")
                continue
                
            # Make prediction
            try:
                X = features_dict[kpi_type]
                
                # Adapt features to match model if needed
                if hasattr(model, 'feature_names_'):
                    from models import adapt_features_to_model
                    X = adapt_features_to_model(model, X)
                
                # Make prediction
                preds = model.predict(X)
                predictions[kpi_type] = float(preds[0])
                
            except Exception as e:
                logger.error(f"Error predicting {kpi_type}: {str(e)}")
                logger.error(traceback.format_exc())
                predictions[kpi_type] = 0.0
        
        return predictions
    
    def evaluate_alternative_content(self, subjects: List[str], preheaders: List[str], 
                                   campaign_metadata: Dict[str, Any],
                                   version: str = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple subject line and preheader combinations
        
        Parameters:
        -----------
        subjects : List[str]
            List of subject line alternatives
        preheaders : List[str]
            List of preheader alternatives
        campaign_metadata : Dict[str, Any]
            Dictionary of campaign metadata
        version : str, optional
            Model version string
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of dictionaries with predictions for each option
        """
        results = []
        
        for i, (subject, preheader) in enumerate(zip(subjects, preheaders)):
            predictions = self.predict_new_campaign(
                subject, preheader, campaign_metadata, version
            )
            
            # Calculate a combined score
            combined_score = self._calculate_combined_score(predictions)
            
            results.append({
                'id': i,
                'subject': subject,
                'preheader': preheader,
                'predictions': predictions,
                'combined_score': combined_score
            })
        
        # Sort by combined score (higher is better)
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def get_best_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the best models for each KPI
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of best model info for each KPI
        """
        return self.best_model_info
    
    def _get_model_params(self, kpi_type: str, model_type: str) -> Dict[str, Any]:
        """
        Get appropriate model parameters based on KPI type and model type
        """
        # Base parameters
        base_params = {
            'random_state': 42
        }
        
        # KPI-specific parameters
        kpi_params = {
            'openrate': {
                'xgboost': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'reg_lambda': 1.0
                },
                'lightgbm': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'reg_lambda': 1.0,
                    'objective': 'regression'
                },
                'catboost': {
                    'iterations': 200,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 1.0
                }
            },
            'clickrate': {
                'xgboost': {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'reg_lambda': 1.5,
                    'min_child_weight': 2
                },
                'lightgbm': {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'reg_lambda': 1.5,
                    'min_child_samples': 5,
                    'objective': 'regression'
                },
                'catboost': {
                    'iterations': 150,
                    'learning_rate': 0.05,
                    'depth': 4,
                    'l2_leaf_reg': 1.5
                }
            },
            'optoutrate': {
                'xgboost': {
                    'n_estimators': 100,
                    'learning_rate': 0.03,
                    'max_depth': 3,
                    'reg_lambda': 2.0,
                    'scale_pos_weight': 5  # Assuming opt-outs are rare events
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'learning_rate': 0.03,
                    'max_depth': 3,
                    'reg_lambda': 2.0,
                    'scale_pos_weight': 5,
                    'objective': 'regression'
                },
                'catboost': {
                    'iterations': 100,
                    'learning_rate': 0.03,
                    'depth': 3,
                    'l2_leaf_reg': 2.0,
                    'scale_pos_weight': 5
                }
            }
        }
        
        # Get KPI-specific parameters
        if kpi_type in kpi_params and model_type in kpi_params[kpi_type]:
            params = {**base_params, **kpi_params[kpi_type][model_type]}
        else:
            # Default parameters
            params = base_params
            
        return params
    
    def _evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, 
                      y_test: pd.Series, kpi_type: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model performance
        
        Returns:
        --------
        float: Primary score for model selection
        Dict[str, float]: Dictionary of performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Select primary score for model selection
        # For now, use R² as the primary metric
        primary_score = r2
        
        return primary_score, metrics
    
    def _calculate_combined_score(self, predictions: Dict[str, float]) -> float:
        """
        Calculate a combined score across KPIs
        
        Optimize: maximize open rate and click rate, minimize opt-out rate
        """
        openrate = predictions.get('openrate', 0)
        clickrate = predictions.get('clickrate', 0)
        optoutrate = predictions.get('optoutrate', 0)
        
        # Simple weighted score
        # Higher is better for opens and clicks, lower is better for opt-outs
        return 0.5 * openrate + 0.5 * clickrate - 2.0 * optoutrate
    
    def _get_model_path(self, kpi_type: str, version: str = None) -> str:
        """Get the file path for a model"""
        model_dir = get_model_directory(version)
        return os.path.join(model_dir, f"{kpi_type}_model.pkl")
    
    def _get_model(self, kpi_type: str, version: str = None) -> Optional[BaseEstimator]:
        """Get a trained model, either from memory or loading from disk"""
        if kpi_type in self.models:
            return self.models[kpi_type]
            
        model_file = self._get_model_path(kpi_type, version)
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                self.models[kpi_type] = model
                return model
            except Exception as e:
                logger.error(f"Error loading model for {kpi_type}: {str(e)}")
                logger.error(traceback.format_exc())
                
        return None
    
    def _save_model(self, model: BaseEstimator, kpi_type: str, 
                  metadata: Dict[str, Any], version: str = None) -> None:
        """Save a model and its metadata"""
        model_dir = get_model_directory(version)
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, f"{kpi_type}_model.pkl")
        
        try:
            # Save model
            joblib.dump(model, model_file)
            
            # Save metadata
            save_model_metadata(metadata, kpi_type, version)
            
            logger.info(f"Saved {kpi_type} model to {model_file}")
        except Exception as e:
            logger.error(f"Error saving {kpi_type} model: {str(e)}")
            logger.error(traceback.format_exc())


class OptimalContentSelector:
    """
    Class for selecting optimal email content (subject lines and preheaders)
    using trained models for all KPIs
    """
    
    def __init__(self, model_manager: MultiKpiModelManager):
        """
        Initialize with a model manager
        """
        self.model_manager = model_manager
    
    def optimize_content(self, 
                        base_subject: str, 
                        base_preheader: str,
                        ai_suggestions: List[Dict[str, str]],
                        campaign_metadata: Dict[str, Any],
                        version: str = None) -> Dict[str, Any]:
        """
        Evaluate and optimize content based on KPI predictions
        
        Parameters:
        -----------
        base_subject : str
            Original subject line
        base_preheader : str
            Original preheader
        ai_suggestions : List[Dict[str, str]]
            List of AI-suggested alternatives with keys 'subject' and 'preheader'
        campaign_metadata : Dict[str, Any]
            Dictionary of campaign metadata
        version : str, optional
            Model version string
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of optimization results
        """
        # Create combined list of all options
        subjects = [base_subject]
        preheaders = [base_preheader]
        
        for suggestion in ai_suggestions:
            subjects.append(suggestion.get('subject', ''))
            preheaders.append(suggestion.get('preheader', ''))
        
        # Evaluate all options
        evaluation_results = self.model_manager.evaluate_alternative_content(
            subjects, preheaders, campaign_metadata, version
        )
        
        # Format results with version labels (A, B, C, D)
        formatted_results = []
        for i, result in enumerate(evaluation_results):
            version_label = chr(65 + i)  # A, B, C, D, ...
            formatted_results.append({
                'version': version_label,
                'subject': result['subject'],
                'preheader': result['preheader'],
                'predictions': result['predictions'],
                'combined_score': result['combined_score']
            })
        
        # Find the best option
        best_option = formatted_results[0]  # Already sorted by combined_score
        
        # Determine if there's an improvement
        baseline = next((r for r in formatted_results if r['version'] == 'A'), None)
        if baseline:
            improvement = {
                'openrate': best_option['predictions']['openrate'] - baseline['predictions']['openrate'],
                'clickrate': best_option['predictions']['clickrate'] - baseline['predictions']['clickrate'],
                'optoutrate': baseline['predictions']['optoutrate'] - best_option['predictions']['optoutrate']
            }
        else:
            improvement = {'openrate': 0, 'clickrate': 0, 'optoutrate': 0}
        
        return {
            'options': formatted_results,
            'best_option': best_option,
            'improvement': improvement
        }