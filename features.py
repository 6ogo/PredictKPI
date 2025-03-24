"""
Feature engineering for KPI Predictor application
"""
import pandas as pd
import numpy as np
import logging
import traceback
import spacy
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import logger, categorize_age

# --- NLP Model Loading ---
def load_nlp_models():
    """Load multilingual models that support Swedish with robust error handling"""
    st_model = None
    nlp_model = None
    
    # Try to load Sentence Transformer model with fallback options
    try:
        # First try with the target model
        try:
            logger.info("Loading SentenceTransformer model paraphrase-multilingual-MiniLM-L12-v2")
            st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            logger.warning(f"Failed to load primary model: {str(e)}")
            # Fallback to a more basic multilingual model
            try:
                logger.info("Trying fallback to distiluse-base-multilingual-cased-v1")
                st_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            except Exception as e2:
                logger.warning(f"Failed to load fallback model: {str(e2)}")
                # Final fallback to the simplest option if everything fails
                logger.info("Using simple word embedding fallback")
                st_model = None
    except ImportError:
        logger.warning("SentenceTransformer library not available")
        st_model = None

    # Load Swedish spaCy model with fallback
    try:
        try:
            nlp_model = spacy.load("sv_core_news_sm")
        except OSError:
            logger.warning("Swedish spaCy model not found. Installing...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "sv_core_news_sm"])
                nlp_model = spacy.load("sv_core_news_sm")
            except Exception as e:
                logger.error(f"Failed to install Swedish spaCy model: {str(e)}")
                # Try using a more common model as fallback
                try:
                    logger.info("Using en_core_web_sm as fallback")
                    nlp_model = spacy.load("en_core_web_sm")
                except:
                    nlp_model = None
    except ImportError:
        logger.warning("spaCy library not available")
        nlp_model = None
    
    return st_model, nlp_model

# --- NLP Feature Extraction ---
def extract_nlp_features(text, st_model, nlp_model):
    """Extract NLP features from text with fallback mechanisms when models aren't available"""
    if not text or pd.isna(text):
        # Return zero vectors for empty texts
        return {
            "embedding": np.zeros(384),  # Size of the embedding vector
            "token_count": 0,
            "has_question": False,
            "exclamation_count": 0,
            "uppercase_ratio": 0.0,
            "punctuation_ratio": 0.0
        }
    
    # Initialize default values
    embedding = np.zeros(384)
    token_count = len(text.split())
    has_question = '?' in text
    exclamation_count = text.count('!')
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    punctuation_ratio = sum(1 for c in text if c in '.,;:!?-()[]{}') / max(len(text), 1)
    
    # Get sentence embedding if model is available
    if st_model is not None:
        try:
            embedding = st_model.encode(text)
        except Exception as e:
            logger.warning(f"Error encoding text with SentenceTransformer: {str(e)}")
            # Create a basic embedding using character counts (very simple fallback)
            unique_chars = set(text.lower())
            char_vec = np.zeros(min(len(unique_chars), 384))
            for i, c in enumerate(unique_chars):
                if i < 384:
                    char_vec[i] = text.lower().count(c) / len(text)
            embedding = char_vec
    
    # Process with spaCy if available
    if nlp_model is not None:
        try:
            doc = nlp_model(text)
            token_count = len([token for token in doc if not token.is_punct])
        except Exception as e:
            logger.warning(f"Error processing text with spaCy: {str(e)}")
    
    return {
        "embedding": embedding, 
        "token_count": token_count,
        "has_question": has_question,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": uppercase_ratio,
        "punctuation_ratio": punctuation_ratio
    }

# --- Vectorized Text Feature Extraction ---
def vectorized_subject_features(subjects):
    """Compute subject line features using vectorized operations"""
    # Convert to string
    subjects = subjects.fillna('').astype(str)
    
    # Basic features
    length = subjects.str.len()
    num_words = subjects.str.split().str.len()
    has_exclamation = subjects.str.contains('!').astype(int)
    has_question = subjects.str.contains(r'\?', regex=True).astype(int)
    
    # Advanced features (vectorized)
    upper_counts = subjects.apply(lambda x: sum(1 for c in x if c.isupper()))
    caps_ratio = upper_counts / length.replace(0, 1)
    
    # Word length calculations
    word_lengths = subjects.apply(lambda x: [len(w) for w in x.split()])
    avg_word_len = word_lengths.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    
    # Special character counts
    special_chars = subjects.apply(lambda x: sum(1 for c in x if c in '!?%$€£#@*&'))
    
    # First and last word length (needs loop)
    first_word_len = subjects.apply(lambda x: len(x.split()[0]) if len(x.split()) > 0 else 0)
    last_word_len = subjects.apply(lambda x: len(x.split()[-1]) if len(x.split()) > 0 else 0)
    
    return pd.DataFrame({
        'length': length,
        'num_words': num_words,
        'has_exclamation': has_exclamation,
        'has_question': has_question,
        'caps_ratio': caps_ratio,
        'avg_word_len': avg_word_len,
        'special_chars': special_chars,
        'first_word_len': first_word_len,
        'last_word_len': last_word_len
    })

# --- Feature Engineering ---
def engineer_features(delivery_data, customer_data, include_preheader=True):
    """
    Perform feature engineering with fallbacks for NLP features
    
    Parameters:
    -----------
    delivery_data : DataFrame
        Delivery data with email campaign information
    customer_data : DataFrame
        Customer demographic data
    include_preheader : bool
        Whether to include preheader features
        
    Returns:
    --------
    dict
        Dictionary of feature DataFrames for different versions
    Series
        Target variable
    dict
        Feature metadata
    """
    try:
        logger.info("Starting feature engineering")
        
        # Calculate KPIs
        delivery_data['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        delivery_data['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
        delivery_data['Optoutrate'] = delivery_data['Optouts'] / delivery_data['Sendouts']
        
        # Aggregate age stats
        age_stats = customer_data.groupby('InternalName')['Age'].agg(['min', 'max']).reset_index()
        age_stats.columns = ['InternalName', 'Min_age', 'Max_age']
        delivery_data = delivery_data.merge(age_stats, on='InternalName', how='left')
        
        # Aggregate Bolag as binary features
        bolag_dummies = pd.get_dummies(customer_data['Bolag'], prefix='Bolag')
        customer_data_with_dummies = pd.concat([customer_data, bolag_dummies], axis=1)
        bolag_features = customer_data_with_dummies.groupby('InternalName')[bolag_dummies.columns].max().reset_index()
        delivery_data = delivery_data.merge(bolag_features, on='InternalName', how='left')
        
        # Try to load NLP models, with graceful fallback
        use_nlp_features = True
        try:
            st_model, nlp_model = load_nlp_models()
            if st_model is None and nlp_model is None:
                logger.warning("Both NLP models failed to load. Using basic features only.")
                use_nlp_features = False
        except Exception as e:
            logger.error(f"Failed to load NLP models: {str(e)}")
            logger.error(traceback.format_exc())
            use_nlp_features = False
            st_model, nlp_model = None, None
        
        # Initialize empty DataFrames for embeddings
        embedding_cols = []
        preheader_embedding_cols = []
        subject_embedding_df = pd.DataFrame(index=delivery_data.index)
        preheader_embedding_df = pd.DataFrame(index=delivery_data.index)
        
        # Process NLP features if models are available
        if use_nlp_features:
            # Process Subject with NLP
            subject_nlp_features = delivery_data['Subject'].apply(
                lambda x: extract_nlp_features(x, st_model, nlp_model)
            )
            
            # Extract NLP embeddings for Subject
            try:
                subject_embeddings = np.vstack([f["embedding"] for f in subject_nlp_features])
                embedding_cols = [f'subject_emb_{i}' for i in range(subject_embeddings.shape[1])]
                subject_embedding_df = pd.DataFrame(
                    subject_embeddings, 
                    columns=embedding_cols,
                    index=delivery_data.index
                )
                
                # Add NLP features for Subject
                for feature in ["token_count", "has_question", "exclamation_count", "uppercase_ratio", "punctuation_ratio"]:
                    delivery_data[f'Subject_nlp_{feature}'] = [f[feature] for f in subject_nlp_features]
                
                # Process Preheader with NLP if included
                if include_preheader:
                    preheader_nlp_features = delivery_data['Preheader'].apply(
                        lambda x: extract_nlp_features(x, st_model, nlp_model)
                    )
                    
                    # Extract NLP embeddings for Preheader
                    preheader_embeddings = np.vstack([f["embedding"] for f in preheader_nlp_features])
                    preheader_embedding_cols = [f'preheader_emb_{i}' for i in range(preheader_embeddings.shape[1])]
                    
                    preheader_embedding_df = pd.DataFrame(
                        preheader_embeddings, 
                        columns=preheader_embedding_cols,
                        index=delivery_data.index
                    )
                    
                    # Add NLP features for Preheader
                    for feature in ["token_count", "has_question", "exclamation_count", "uppercase_ratio", "punctuation_ratio"]:
                        delivery_data[f'Preheader_nlp_{feature}'] = [f[feature] for f in preheader_nlp_features]
            except Exception as e:
                logger.error(f"Error processing embeddings: {str(e)}")
                logger.error(traceback.format_exc())
                use_nlp_features = False
        
        # If NLP features couldn't be processed, create dummy columns
        if not use_nlp_features:
            logger.info("Adding placeholder NLP feature columns")
            # Add empty NLP features for Subject
            for feature in ["token_count", "has_question", "exclamation_count", "uppercase_ratio", "punctuation_ratio"]:
                delivery_data[f'Subject_nlp_{feature}'] = 0
            
            # Add empty NLP features for Preheader if needed
            if include_preheader:
                for feature in ["token_count", "has_question", "exclamation_count", "uppercase_ratio", "punctuation_ratio"]:
                    delivery_data[f'Preheader_nlp_{feature}'] = 0
        
        # Basic text features for subject (vectorized)
        subject_features = vectorized_subject_features(delivery_data['Subject'])
        delivery_data['Subject_length'] = subject_features['length']
        delivery_data['Num_words'] = subject_features['num_words']
        delivery_data['Has_exclamation'] = subject_features['has_exclamation']
        delivery_data['Has_question'] = subject_features['has_question']
        
        # Enhanced text features for subject (v2.x+)
        delivery_data['Subject_num_words'] = subject_features['num_words']
        delivery_data['Subject_has_exclamation'] = subject_features['has_exclamation']
        delivery_data['Subject_has_question'] = subject_features['has_question']
        delivery_data['Subject_caps_ratio'] = subject_features['caps_ratio']
        delivery_data['Subject_avg_word_len'] = subject_features['avg_word_len']
        delivery_data['Subject_num_special_chars'] = subject_features['special_chars']
        delivery_data['Subject_first_word_len'] = subject_features['first_word_len']
        delivery_data['Subject_last_word_len'] = subject_features['last_word_len']
        
        # Add preheader features if requested (for version 2.x)
        if include_preheader:
            # Use the same vectorized approach for preheader
            preheader_features = vectorized_subject_features(delivery_data['Preheader'])
            
            # Basic preheader features
            delivery_data['Preheader_length'] = preheader_features['length']
            delivery_data['Preheader_num_words'] = preheader_features['num_words']
            delivery_data['Preheader_has_exclamation'] = preheader_features['has_exclamation']
            delivery_data['Preheader_has_question'] = preheader_features['has_question']
            
            # Enhanced preheader features
            delivery_data['Preheader_caps_ratio'] = preheader_features['caps_ratio']
            delivery_data['Preheader_avg_word_len'] = preheader_features['avg_word_len']
            delivery_data['Preheader_num_special_chars'] = preheader_features['special_chars']
            
            # Relationship between subject and preheader
            delivery_data['Subject_preheader_length_ratio'] = delivery_data['Subject_length'] / delivery_data['Preheader_length'].replace(0, 1)
            delivery_data['Subject_preheader_words_ratio'] = delivery_data['Subject_num_words'] / delivery_data['Preheader_num_words'].replace(0, 1)
        
        # Define feature columns
        categorical_features = ['Dialog', 'Syfte', 'Product']
        
        # Define numerical features based on version
        legacy_numerical_features = [
            'Min_age', 'Max_age', 
            'Subject_length', 'Num_words', 'Has_exclamation', 'Has_question',
            # Add NLP features to legacy
            'Subject_nlp_token_count', 'Subject_nlp_has_question', 
            'Subject_nlp_exclamation_count', 'Subject_nlp_uppercase_ratio', 
            'Subject_nlp_punctuation_ratio'
        ]
        
        v2_numerical_features = [
            'Min_age', 'Max_age', 
            'Subject_length', 'Subject_num_words', 'Subject_has_exclamation', 'Subject_has_question',
            'Subject_caps_ratio', 'Subject_avg_word_len', 'Subject_num_special_chars',
            'Subject_first_word_len', 'Subject_last_word_len',
            # Add NLP features to v2
            'Subject_nlp_token_count', 'Subject_nlp_has_question', 
            'Subject_nlp_exclamation_count', 'Subject_nlp_uppercase_ratio', 
            'Subject_nlp_punctuation_ratio'
        ]
        
        if include_preheader:
            v2_numerical_features.extend([
                'Preheader_length', 'Preheader_num_words', 
                'Preheader_has_exclamation', 'Preheader_has_question',
                'Preheader_caps_ratio', 'Preheader_avg_word_len', 'Preheader_num_special_chars',
                'Subject_preheader_length_ratio', 'Subject_preheader_words_ratio',
                # Add NLP features for preheader
                'Preheader_nlp_token_count', 'Preheader_nlp_has_question', 
                'Preheader_nlp_exclamation_count', 'Preheader_nlp_uppercase_ratio', 
                'Preheader_nlp_punctuation_ratio'
            ])
        
        bolag_features_list = [col for col in delivery_data.columns if col.startswith('Bolag_')]
        
        # Track feature sets for different model versions
        feature_sets = {
            'legacy': {
                'categorical': categorical_features,
                'numerical': legacy_numerical_features,
                'bolag': bolag_features_list,
                'embeddings': embedding_cols
            },
            'v2': {
                'categorical': categorical_features,
                'numerical': v2_numerical_features,
                'bolag': bolag_features_list,
                'embeddings': embedding_cols + (preheader_embedding_cols if include_preheader else [])
            }
        }
        
        # Generate categorical dummies
        dummy_df = pd.get_dummies(delivery_data[categorical_features])
        
        # Create mappings for UI
        dummy_dialog_map = {dialog: f'Dialog_{dialog}' for dialog in delivery_data['Dialog'].unique()}
        dummy_syfte_map = {syfte: f'Syfte_{syfte}' for syfte in delivery_data['Syfte'].unique()}
        dummy_product_map = {product: f'Product_{product}' for product in delivery_data['Product'].unique()}
        
        # Prepare features for legacy model (v1.x) with embeddings
        legacy_features = pd.concat([
            dummy_df,
            delivery_data[legacy_numerical_features],
            delivery_data[bolag_features_list].fillna(0).astype(int),
            subject_embedding_df  # Add embeddings
        ], axis=1)
        
        # Prepare features for v2 model with embeddings
        if include_preheader:
            v2_features = pd.concat([
                dummy_df,
                delivery_data[v2_numerical_features],
                delivery_data[bolag_features_list].fillna(0).astype(int),
                subject_embedding_df,  # Add subject embeddings
                preheader_embedding_df  # Add preheader embeddings
            ], axis=1)
        else:
            v2_features = pd.concat([
                dummy_df,
                delivery_data[v2_numerical_features],
                delivery_data[bolag_features_list].fillna(0).astype(int),
                subject_embedding_df  # Add only subject embeddings
            ], axis=1)
        
        # Target variable
        target = delivery_data['Openrate']
        
        # Metadata for documentation
        feature_metadata = {
            'categorical_features': categorical_features,
            'legacy_numerical_features': legacy_numerical_features,
            'v2_numerical_features': v2_numerical_features,
            'bolag_features': bolag_features_list,
            'subject_embedding_features': embedding_cols,
            'preheader_embedding_features': preheader_embedding_cols if include_preheader else [],
            'dummy_dialog_map': dummy_dialog_map,
            'dummy_syfte_map': dummy_syfte_map,
            'dummy_product_map': dummy_product_map,
            'feature_sets': feature_sets,
            'include_preheader': include_preheader,
            'nlp_model_used': 'paraphrase-multilingual-MiniLM-L12-v2 (with fallbacks)'
        }
        
        logger.info(f"Feature engineering completed - Legacy features: {legacy_features.shape}, V2 features: {v2_features.shape}")
        
        return {
            'legacy': legacy_features, 
            'v2': v2_features
        }, target, feature_metadata
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Enhanced Feature Engineering ---
def enhanced_feature_engineering(delivery_data, customer_data, include_preheader=True):
    """
    Extend the feature set with more advanced features
    
    Parameters:
    -----------
    delivery_data : DataFrame
        Delivery data with email campaign information
    customer_data : DataFrame
        Customer demographic data
    include_preheader : bool
        Whether to include preheader features
        
    Returns:
    --------
    tuple
        Enhanced features, target, and feature metadata
    """
    try:
        # First get basic features
        features_dict, target, feature_metadata = engineer_features(
            delivery_data, customer_data, 
            include_preheader=include_preheader
        )
        
        # Adding temporal features
        if 'SendDate' in delivery_data.columns:
            logger.info("Adding temporal features")
            
            # Convert to datetime
            delivery_data['SendDate'] = pd.to_datetime(delivery_data['SendDate'], errors='coerce')
            
            # Extract temporal components
            delivery_data['SendMonth'] = delivery_data['SendDate'].dt.month
            delivery_data['SendDayOfWeek'] = delivery_data['SendDate'].dt.dayofweek
            delivery_data['SendHour'] = delivery_data['SendDate'].dt.hour
            delivery_data['IsWeekend'] = delivery_data['SendDayOfWeek'].isin([5, 6]).astype(int)
            
            # Create month and day of week dummies
            month_dummies = pd.get_dummies(delivery_data['SendMonth'], prefix='Month')
            dow_dummies = pd.get_dummies(delivery_data['SendDayOfWeek'], prefix='DayOfWeek')
            
            # Add to feature sets
            for version in features_dict:
                features_dict[version] = pd.concat([
                    features_dict[version],
                    month_dummies,
                    dow_dummies,
                    delivery_data[['SendHour', 'IsWeekend']]
                ], axis=1)
                
            # Update metadata
            feature_metadata['temporal_features'] = {
                'month_features': month_dummies.columns.tolist(),
                'day_of_week_features': dow_dummies.columns.tolist(),
                'other_features': ['SendHour', 'IsWeekend']
            }
        
        # Add TF-IDF features for subject lines
        try:
            # Configure vectorizer for Swedish text
            vectorizer = TfidfVectorizer(
                max_features=50,  # Limit to top 50 features to avoid dimensionality issues
                min_df=5,        # Minimum document frequency
                max_df=0.7,      # Maximum document frequency
                stop_words=None  # No built-in Swedish stopwords, consider providing a custom list
            )
            
            # Fit and transform the subject lines
            tfidf_subject = vectorizer.fit_transform(delivery_data['Subject'].fillna(''))
            
            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_subject.toarray(),
                columns=[f'subject_tfidf_{i}' for i in range(tfidf_subject.shape[1])],
                index=delivery_data.index
            )
            
            # Add to feature sets
            for version in features_dict:
                features_dict[version] = pd.concat([
                    features_dict[version],
                    tfidf_df
                ], axis=1)
                
            # Add preheader TF-IDF if included
            if include_preheader:
                # Fit and transform the preheaders
                tfidf_preheader = vectorizer.fit_transform(delivery_data['Preheader'].fillna(''))
                
                # Convert to DataFrame
                tfidf_preheader_df = pd.DataFrame(
                    tfidf_preheader.toarray(),
                    columns=[f'preheader_tfidf_{i}' for i in range(tfidf_preheader.shape[1])],
                    index=delivery_data.index
                )
                
                # Add to feature sets
                for version in features_dict:
                    features_dict[version] = pd.concat([
                        features_dict[version],
                        tfidf_preheader_df
                    ], axis=1)
                
            # Update metadata
            feature_metadata['tfidf_features'] = {
                'subject_features': tfidf_df.columns.tolist(),
                'preheader_features': tfidf_preheader_df.columns.tolist() if include_preheader else []
            }
            
        except Exception as e:
            logger.warning(f"Could not add TF-IDF features: {str(e)}")
        
        # Add interaction features
        logger.info("Adding interaction features")
        for version in features_dict:
            features = features_dict[version]
            
            # Interaction between dialog and product
            dialog_columns = [col for col in features.columns if col.startswith('Dialog_')]
            product_columns = [col for col in features.columns if col.startswith('Product_')]
            
            # Limit interactions to avoid feature explosion
            top_dialogs = dialog_columns[:min(5, len(dialog_columns))]
            top_products = product_columns[:min(5, len(product_columns))]
            
            for dialog_col in top_dialogs:
                for product_col in top_products:
                    interaction_name = f"Interact_{dialog_col}_{product_col}"
                    features[interaction_name] = features[dialog_col] * features[product_col]
            
            # Interaction between age and other features
            if 'Min_age' in features.columns and 'Max_age' in features.columns:
                # Average age
                features['Avg_age'] = (features['Min_age'] + features['Max_age']) / 2
                
                # Age interactions with key features
                for feature in ['Subject_length', 'Subject_has_question']:
                    if feature in features.columns:
                        features[f'Age_{feature}'] = features['Avg_age'] * features[feature]
        
        # Update metadata
        feature_metadata['added_interactions'] = True
        
        logger.info(f"Enhanced feature engineering completed")
        return features_dict, target, feature_metadata
    
    except Exception as e:
        logger.error(f"Error in enhanced feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to basic features
        return engineer_features(delivery_data, customer_data, include_preheader)

# --- Feature Selection ---
def select_features(X, y, max_features=100, method='importance'):
    """
    Select the most important features to reduce dimensionality
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    max_features : int
        Maximum number of features to select
    method : str
        Method to use ('importance', 'mutual_info', 'recursive')
        
    Returns:
    --------
    DataFrame
        Selected features
    list
        Names of selected features
    """
    try:
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
        from xgboost import XGBRegressor
        
        # If we already have fewer features than max_features, return all
        if X.shape[1] <= max_features:
            return X, X.columns.tolist()
        
        if method == 'importance':
            # Select using XGBoost feature importance
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get importance scores
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = importance['feature'].head(max_features).tolist()
            
        elif method == 'mutual_info':
            # Select using mutual information
            selector = SelectKBest(mutual_info_regression, k=max_features)
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [X.columns[i] for i in selected_indices]
            
        elif method == 'recursive':
            # Recursive feature elimination
            estimator = XGBRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=max_features, step=0.1)
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_features = [X.columns[i] for i, selected in enumerate(selector.support_) if selected]
            
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using all features.")
            return X, X.columns.tolist()
        
        # Return selected features
        X_selected = X[selected_features]
        return X_selected, selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to all features
        return X, X.columns.tolist()

# --- Age Group Analysis ---
def process_data_for_age_heatmap(delivery_data, customer_data):
    """
    Process data for age group heatmap
    """
    try:
        logger.info("Processing data for age group heatmap")
        
        # Add age group to customer data
        customer_data['AgeGroup'] = customer_data['Age'].apply(categorize_age)
        
        # Merge customer data with delivery data
        merged_data = delivery_data.merge(customer_data, on='InternalName', how='left')
        
        # Ensure we have all age groups (even if zero data)
        all_age_groups = ['18-24', '25-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
        
        # Calculate rates by age group
        results = []
        
        # Create a dictionary to store the aggregated results by age group and product, dialog, or syfte
        agg_by_dialog = {}
        agg_by_syfte = {}
        agg_by_product = {}
        
        # Process by age group
        for age_group in all_age_groups:
            age_data = merged_data[merged_data['AgeGroup'] == age_group]
            
            if len(age_data) == 0:
                # Add a row with zeros if no data for this age group
                results.append({
                    'AgeGroup': age_group,
                    'Openrate': 0,
                    'Clickrate': 0,
                    'Optoutrate': 0,
                    'Count': 0
                })
            else:
                # Calculate overall metrics
                total_sendouts = age_data['Sendouts'].sum()
                total_opens = age_data['Opens'].sum()
                total_clicks = age_data['Clicks'].sum()
                total_optouts = age_data['Optouts'].sum()
                
                openrate = total_opens / total_sendouts if total_sendouts > 0 else 0
                clickrate = total_clicks / total_sendouts if total_sendouts > 0 else 0
                optoutrate = total_optouts / total_sendouts if total_sendouts > 0 else 0
                
                results.append({
                    'AgeGroup': age_group,
                    'Openrate': openrate,
                    'Clickrate': clickrate,
                    'Optoutrate': optoutrate,
                    'Count': len(age_data)
                })
                
                # Aggregate by Dialog
                for dialog in age_data['Dialog'].unique():
                    dialog_data = age_data[age_data['Dialog'] == dialog]
                    dialog_sendouts = dialog_data['Sendouts'].sum()
                    dialog_opens = dialog_data['Opens'].sum()
                    dialog_clicks = dialog_data['Clicks'].sum()
                    dialog_optouts = dialog_data['Optouts'].sum()
                    
                    if dialog not in agg_by_dialog:
                        agg_by_dialog[dialog] = {}
                    
                    agg_by_dialog[dialog][age_group] = {
                        'Openrate': dialog_opens / dialog_sendouts if dialog_sendouts > 0 else 0,
                        'Clickrate': dialog_clicks / dialog_sendouts if dialog_sendouts > 0 else 0,
                        'Optoutrate': dialog_optouts / dialog_sendouts if dialog_sendouts > 0 else 0,
                        'Count': len(dialog_data)
                    }
                
                # Aggregate by Syfte
                for syfte in age_data['Syfte'].unique():
                    syfte_data = age_data[age_data['Syfte'] == syfte]
                    syfte_sendouts = syfte_data['Sendouts'].sum()
                    syfte_opens = syfte_data['Opens'].sum()
                    syfte_clicks = syfte_data['Clicks'].sum()
                    syfte_optouts = syfte_data['Optouts'].sum()
                    
                    if syfte not in agg_by_syfte:
                        agg_by_syfte[syfte] = {}
                    
                    agg_by_syfte[syfte][age_group] = {
                        'Openrate': syfte_opens / syfte_sendouts if syfte_sendouts > 0 else 0,
                        'Clickrate': syfte_clicks / syfte_sendouts if syfte_sendouts > 0 else 0,
                        'Optoutrate': syfte_optouts / syfte_sendouts if syfte_sendouts > 0 else 0,
                        'Count': len(syfte_data)
                    }
                
                # Aggregate by Product
                for product in age_data['Product'].unique():
                    product_data = age_data[age_data['Product'] == product]
                    product_sendouts = product_data['Sendouts'].sum()
                    product_opens = product_data['Opens'].sum()
                    product_clicks = product_data['Clicks'].sum()
                    product_optouts = product_data['Optouts'].sum()
                    
                    if product not in agg_by_product:
                        agg_by_product[product] = {}
                    
                    agg_by_product[product][age_group] = {
                        'Openrate': product_opens / product_sendouts if product_sendouts > 0 else 0,
                        'Clickrate': product_clicks / product_sendouts if product_sendouts > 0 else 0,
                        'Optoutrate': product_optouts / product_sendouts if product_sendouts > 0 else 0,
                        'Count': len(product_data)
                    }
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create a uniform DataFrame for each metric
        metrics = ['Openrate', 'Clickrate', 'Optoutrate']
        heatmap_data = {metric: pd.DataFrame(index=all_age_groups) for metric in metrics}
        
        # Prepare for overall heatmap
        for metric in metrics:
            heatmap_data[metric]['Overall'] = results_df.set_index('AgeGroup')[metric]
        
        # Add dialog, syfte, and product information
        from utils import DIALOG_VALUES, SYFTE_VALUES, PRODUKT_VALUES
        
        for dialog, data in agg_by_dialog.items():
            for metric in metrics:
                dialog_values = [data.get(age_group, {}).get(metric, 0) for age_group in all_age_groups]
                display_dialog = next((label for code, label in DIALOG_VALUES.items() if code[0] == dialog), dialog)
                heatmap_data[metric][f"Dialog: {display_dialog}"] = dialog_values
        
        for syfte, data in agg_by_syfte.items():
            for metric in metrics:
                syfte_values = [data.get(age_group, {}).get(metric, 0) for age_group in all_age_groups]
                display_syfte = next((label for code, label in SYFTE_VALUES.items() if code[0] == syfte), syfte)
                heatmap_data[metric][f"Syfte: {display_syfte}"] = syfte_values
        
        for product, data in agg_by_product.items():
            for metric in metrics:
                product_values = [data.get(age_group, {}).get(metric, 0) for age_group in all_age_groups]
                display_product = next((label for code, label in PRODUKT_VALUES.items() if code[0] == product), product)
                heatmap_data[metric][f"Product: {display_product}"] = product_values
        
        return heatmap_data, results_df
    except Exception as e:
        logger.error(f"Error processing data for age heatmap: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def prepare_version_heatmap_data(all_options, heatmap_data, metric):
    """
    Prepare data for version comparison heatmap with proportional estimates for different versions
    
    Parameters:
    -----------
    all_options : list
        List of tuples (version, subject, preheader, openrate)
    heatmap_data : dict
        Dictionary of heatmap data by metric
    metric : str
        Metric to use ('Openrate', 'Clickrate', 'Optoutrate')
    
    Returns:
    --------
    DataFrame
        DataFrame with age groups as index and versions as columns
    """
    # Get base data and age groups
    base_data = heatmap_data[metric].copy()
    age_groups = base_data.index
    
    # Create new DataFrame with just age groups
    version_data = pd.DataFrame(index=age_groups)
    
    # Get the baseline data and the baseline overall rate
    baseline_values = base_data['Overall'].values
    
    # For openrate: use the predicted values from the model
    if metric == 'Openrate':
        # Find the baseline (version A) open rate
        baseline_overall = next(rate for ver, _, _, rate in all_options if ver == 'A')
        
        for version, _, _, predicted_rate in all_options:
            # Calculate the ratio between this version's predicted rate and the baseline
            if baseline_overall > 0:
                adjustment_ratio = predicted_rate / baseline_overall
            else:
                adjustment_ratio = 1.0
                
            # Apply this ratio to adjust each age group's rate
            # Use numpy's clip to ensure values stay in reasonable range (0-100%)
            adjusted_values = np.clip(baseline_values * adjustment_ratio, 0, 1)
            
            # Add to dataframe
            version_data[f"Version {version}"] = adjusted_values
            
    # For clickrate and optoutrate: simulate effect based on open rate change
    else:
        # Find baseline values
        baseline_overall = next(rate for ver, _, _, rate in all_options if ver == 'A')
        
        for version, _, _, predicted_rate in all_options:
            # Calculate modification based on open rate change (simplified model)
            # Assumption: as open rate increases, click rate increases proportionally but less dramatically
            # and optout rate decreases slightly
            if baseline_overall > 0:
                ratio = predicted_rate / baseline_overall
                
                if metric == 'Clickrate':
                    # Click rate increases with open rate but with diminishing returns
                    adjustment_ratio = 1.0 + (ratio - 1.0) * 0.7
                else:  # Optoutrate
                    # Optout rate slightly decreases as open rate increases (inverse relationship)
                    adjustment_ratio = 1.0 - (ratio - 1.0) * 0.3
            else:
                adjustment_ratio = 1.0
                
            # Apply adjustment
            adjusted_values = np.clip(baseline_values * adjustment_ratio, 0, 1)
            
            # Add to dataframe
            version_data[f"Version {version}"] = adjusted_values
    
    return version_data