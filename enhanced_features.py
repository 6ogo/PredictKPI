"""
Enhanced feature engineering for multi-KPI prediction
"""
import pandas as pd
import numpy as np
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import logger, categorize_age


def prepare_multi_kpi_features(delivery_data: pd.DataFrame, customer_data: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """
    Prepare features and targets for multiple KPI models
    
    Parameters:
    -----------
    delivery_data : DataFrame
        Delivery data with email campaign information
    customer_data : DataFrame
        Customer demographic data
        
    Returns:
    --------
    Tuple[Dict[str, DataFrame], Dict[str, Series]]
        Dictionary of features and targets for each KPI type
    """
    try:
        logger.info("Preparing features for multiple KPI prediction")
        
        # Ensure Preheader column exists
        if 'Preheader' not in delivery_data.columns:
            logger.warning("Preheader column not found in delivery data. Adding empty column.")
            delivery_data['Preheader'] = ''
        
        # Calculate KPIs
        delivery_data['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        delivery_data['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
        delivery_data['Optoutrate'] = delivery_data['Optouts'] / delivery_data['Sendouts']
        
        # Create feature sets for each KPI
        features_dict = {}
        target_dict = {}
        
        # --- 1. Features for open rate prediction ---
        openrate_features, openrate_target = engineer_openrate_features(delivery_data, customer_data)
        features_dict['openrate'] = openrate_features
        target_dict['openrate'] = openrate_target
        
        # --- 2. Features for click rate prediction ---
        clickrate_features, clickrate_target = engineer_clickrate_features(delivery_data, customer_data)
        features_dict['clickrate'] = clickrate_features
        target_dict['clickrate'] = clickrate_target
        
        # --- 3. Features for opt-out rate prediction ---
        optoutrate_features, optoutrate_target = engineer_optoutrate_features(delivery_data, customer_data)
        features_dict['optoutrate'] = optoutrate_features
        target_dict['optoutrate'] = optoutrate_target
        
        return features_dict, target_dict
        
    except Exception as e:
        logger.error(f"Error preparing multi-KPI features: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def engineer_openrate_features(delivery_data: pd.DataFrame, customer_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features specifically for open rate prediction
    """
    try:
        # Start with common features
        features, _ = engineer_common_features(delivery_data, customer_data)
        
        # Add open rate specific features
        
        # 1. Subject line specifics highly relevant for opens
        features['Subject_first_word_is_question'] = delivery_data['Subject'].str.split().str[0].str.contains(r'\?').fillna(False).astype(int)
        features['Subject_contains_number'] = delivery_data['Subject'].str.contains(r'\d').fillna(False).astype(int)
        features['Subject_contains_personalization'] = delivery_data['Subject'].str.contains('{{').fillna(False).astype(int)
        
        # 2. Preheader characteristics - Fix the problematic line
        # Create a function to check if preheader complements subject
        def preheader_complements_subject(row):
            if pd.isna(row['Subject']) or pd.isna(row['Preheader']) or not row['Preheader']:
                return 0
            preheader_start = row['Preheader'][:20] if len(row['Preheader']) >= 20 else row['Preheader']
            return 1 if preheader_start not in row['Subject'] and len(row['Preheader']) > 10 else 0
        
        # Apply the function to each row
        features['Preheader_complements_subject'] = delivery_data.apply(preheader_complements_subject, axis=1)
        
        # 3. Time-based features
        if 'Date' in delivery_data.columns:
            delivery_data['Date'] = pd.to_datetime(delivery_data['Date'], errors='coerce')
            features['Send_hour'] = delivery_data['Date'].dt.hour
            features['Send_day_of_week'] = delivery_data['Date'].dt.dayofweek
            features['Is_weekend'] = delivery_data['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Hour categories (morning, afternoon, evening)
            features['Send_morning'] = (delivery_data['Date'].dt.hour.between(6, 11)).astype(int)
            features['Send_afternoon'] = (delivery_data['Date'].dt.hour.between(12, 17)).astype(int)
            features['Send_evening'] = (delivery_data['Date'].dt.hour.between(18, 23)).astype(int)
        else:
            # Handle the case when Date column is missing
            logger.warning("Date column missing, skipping time-based features")
            features['Send_hour'] = 0
            features['Send_day_of_week'] = 0
            features['Is_weekend'] = 0
            features['Send_morning'] = 0
            features['Send_afternoon'] = 0
            features['Send_evening'] = 0
        
        # Return features and target
        return features, delivery_data['Openrate']
        
    except Exception as e:
        logger.error(f"Error engineering open rate features: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def engineer_clickrate_features(delivery_data: pd.DataFrame, customer_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features specifically for click rate prediction
    """
    try:
        # Start with common features
        features, _ = engineer_common_features(delivery_data, customer_data)
        
        # Add click rate specific features
        
        # 1. Include open rate as a feature (clicks depend on opens)
        features['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        
        # 2. Add features related to call-to-action
        features['Subject_has_cta'] = delivery_data['Subject'].str.contains('klicka|läs mer|se här|upptäck', case=False).fillna(False).astype(int)
        features['Preheader_has_cta'] = delivery_data['Preheader'].str.contains('klicka|läs mer|se här|upptäck', case=False).fillna(False).astype(int)
        
        # 3. Content value indicators
        features['Subject_suggests_value'] = delivery_data['Subject'].str.contains('rabatt|erbjudande|spara|ny|exklusiv', case=False).fillna(False).astype(int)
        features['Preheader_suggests_value'] = delivery_data['Preheader'].str.contains('rabatt|erbjudande|spara|ny|exklusiv', case=False).fillna(False).astype(int)
        
        # Return features and target
        return features, delivery_data['Clickrate']
        
    except Exception as e:
        logger.error(f"Error engineering click rate features: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def engineer_optoutrate_features(delivery_data: pd.DataFrame, customer_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features specifically for opt-out rate prediction
    """
    try:
        # Start with common features
        features, _ = engineer_common_features(delivery_data, customer_data)
        
        # Add opt-out rate specific features
        
        # 1. Include engagement metrics
        features['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        features['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
        
        # 2. Add features related to potentially annoying content
        features['Subject_all_caps'] = delivery_data['Subject'].str.isupper().fillna(False).astype(int)
        features['Subject_has_urgency'] = delivery_data['Subject'].str.contains('nu|idag|sista chansen|missa inte', case=False).fillna(False).astype(int)
        features['Subject_many_exclamations'] = (delivery_data['Subject'].str.count('!') > 1).fillna(False).astype(int)
        
        # 3. Message frequency indicators (if available)
        # This would require tracking previous messages to each recipient
        
        # Return features and target
        return features, delivery_data['Optoutrate']
        
    except Exception as e:
        logger.error(f"Error engineering opt-out rate features: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def engineer_common_features(delivery_data: pd.DataFrame, customer_data: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    """
    Engineer common features for all KPI models
    """
    try:
        # --- Text features ---
        # Subject features
        subject_features = vectorized_subject_features(delivery_data['Subject'])
        
        # Preheader features
        preheader_features = vectorized_subject_features(delivery_data['Preheader'].fillna(''))
        
        # --- Categorical features ---
        # One-hot encode categorical variables
        dialog_dummies = pd.get_dummies(delivery_data['Dialog'], prefix='Dialog')
        syfte_dummies = pd.get_dummies(delivery_data['Syfte'], prefix='Syfte')
        product_dummies = pd.get_dummies(delivery_data['Product'], prefix='Product')
        
        # --- Audience features ---
        # Aggregate age stats
        age_stats = customer_data.groupby('InternalName')['Age'].agg(['min', 'max', 'mean']).reset_index()
        age_stats.columns = ['InternalName', 'Min_age', 'Max_age', 'Mean_age']
        
        # Merge age stats with delivery data
        delivery_with_age = delivery_data.merge(age_stats, on='InternalName', how='left')
        
        # --- Age group distribution ---
        # Add age group to customer data
        customer_data['AgeGroup'] = customer_data['Age'].apply(categorize_age)
        
        # Calculate age group distribution for each campaign
        age_group_dummies = pd.get_dummies(customer_data['AgeGroup'], prefix='AgeGroup')
        customer_with_age_dummies = pd.concat([customer_data[['InternalName']], age_group_dummies], axis=1)
        
        age_dist = customer_with_age_dummies.groupby('InternalName').mean().reset_index()
        
        # Merge age distribution with delivery data
        delivery_with_age_dist = delivery_with_age.merge(age_dist, on='InternalName', how='left')
        
        # --- Gender distribution ---
        # Calculate gender distribution for each campaign
        gender_dummies = pd.get_dummies(customer_data['Gender'], prefix='Gender')
        customer_with_gender_dummies = pd.concat([customer_data[['InternalName']], gender_dummies], axis=1)
        
        gender_dist = customer_with_gender_dummies.groupby('InternalName').mean().reset_index()
        
        # Merge gender distribution with delivery data
        delivery_with_demographics = delivery_with_age_dist.merge(gender_dist, on='InternalName', how='left')
        
        # --- Bolag distribution ---
        # Calculate Bolag distribution for each campaign
        bolag_dummies = pd.get_dummies(customer_data['Bolag'], prefix='Bolag')
        customer_with_bolag_dummies = pd.concat([customer_data[['InternalName']], bolag_dummies], axis=1)
        
        bolag_dist = customer_with_bolag_dummies.groupby('InternalName').mean().reset_index()
        
        # Merge Bolag distribution with delivery data
        delivery_with_all = delivery_with_demographics.merge(bolag_dist, on='InternalName', how='left')
        
        # --- Combine all features ---
        # Rename subject features
        subject_features_renamed = subject_features.add_prefix('Subject_')
        
        # Rename preheader features
        preheader_features_renamed = preheader_features.add_prefix('Preheader_')
        
        # Select demographic features
        demographic_features = ['Min_age', 'Max_age', 'Mean_age']
        
        # Age group columns
        age_group_cols = [col for col in delivery_with_all.columns if col.startswith('AgeGroup_')]
        
        # Gender columns
        gender_cols = [col for col in delivery_with_all.columns if col.startswith('Gender_')]
        
        # Bolag columns
        bolag_cols = [col for col in delivery_with_all.columns if col.startswith('Bolag_')]
        
        # All demographic columns
        all_demographic_cols = demographic_features + age_group_cols + gender_cols + bolag_cols
        
        # Create dataframe with all features
        features = pd.DataFrame(index=delivery_data.index)
        
        # Add subject features
        for col in subject_features_renamed.columns:
            features[col] = subject_features_renamed[col].values
        
        # Add preheader features
        for col in preheader_features_renamed.columns:
            features[col] = preheader_features_renamed[col].values
        
        # Add categorical features
        for df in [dialog_dummies, syfte_dummies, product_dummies]:
            for col in df.columns:
                features[col] = df[col].values
        
        # Add demographic features
        for col in all_demographic_cols:
            features[col] = delivery_with_all[col].values
        
        # Add relationship between subject and preheader
        features['Subject_preheader_length_ratio'] = (
            delivery_data['Subject'].str.len() / 
            delivery_data['Preheader'].str.len().replace(0, 1)
        ).fillna(1)
        
        # Calculate text similarity between subject and preheader
        features['Subject_preheader_word_overlap'] = calculate_word_overlap(
            delivery_data['Subject'], delivery_data['Preheader']
        )
        
        # Fill NA values
        features = features.fillna(0)
        
        return features, None
        
    except Exception as e:
        logger.error(f"Error engineering common features: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def vectorized_subject_features(texts):
    """Compute text features using vectorized operations"""
    # Convert to string
    texts = texts.fillna('').astype(str)
    
    # Basic features
    length = texts.str.len()
    num_words = texts.str.split().str.len()
    has_exclamation = texts.str.contains('!').astype(int)
    has_question = texts.str.contains(r'\?', regex=True).astype(int)  # Fixed escape sequence
    
    # Advanced features (vectorized)
    upper_counts = texts.apply(lambda x: sum(1 for c in x if c.isupper()))
    caps_ratio = upper_counts / length.replace(0, 1)
    
    # Word length calculations
    word_lengths = texts.apply(lambda x: [len(w) for w in x.split()])
    avg_word_len = word_lengths.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    
    # Special character counts
    special_chars = texts.apply(lambda x: sum(1 for c in x if c in '!?%$€£#@*&'))
    
    # First and last word length
    first_word_len = texts.apply(lambda x: len(x.split()[0]) if len(x.split()) > 0 else 0)
    last_word_len = texts.apply(lambda x: len(x.split()[-1]) if len(x.split()) > 0 else 0)
    
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


def calculate_word_overlap(series1, series2):
    """Calculate word overlap between two text series"""
    def word_overlap(text1, text2):
        if pd.isna(text1) or pd.isna(text2) or not text1 or not text2:
            return 0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    return pd.Series([word_overlap(t1, t2) for t1, t2 in zip(series1, series2)])


def create_campaign_level_features(subject: str, preheader: str, campaign_metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Create feature matrices for a new campaign for all KPI types
    
    Parameters:
    -----------
    subject : str
        Email subject line
    preheader : str
        Email preheader
    campaign_metadata : Dict[str, Any]
        Campaign metadata including dialog, syfte, product, age range, etc.
        
    Returns:
    --------
    Dict[str, DataFrame]
        Dictionary of feature DataFrames for each KPI type
    """
    try:
        # Create a DataFrame with one row
        df = pd.DataFrame({
            'Subject': [subject],
            'Preheader': [preheader],
            'Dialog': [campaign_metadata.get('dialog', '')],
            'Syfte': [campaign_metadata.get('syfte', '')],
            'Product': [campaign_metadata.get('product', '')]
        })
        
        # Add common text features
        subject_features = vectorized_subject_features(df['Subject'])
        subject_features_renamed = subject_features.add_prefix('Subject_')
        
        preheader_features = vectorized_subject_features(df['Preheader'])
        preheader_features_renamed = preheader_features.add_prefix('Preheader_')
        
        # Create one-hot encodings for categorical features
        dialog_dummies = pd.get_dummies(df['Dialog'], prefix='Dialog')
        syfte_dummies = pd.get_dummies(df['Syfte'], prefix='Syfte')
        product_dummies = pd.get_dummies(df['Product'], prefix='Product')
        
        # Combine features
        features = pd.DataFrame(index=[0])
        
        # Add text features
        for col in subject_features_renamed.columns:
            features[col] = subject_features_renamed[col].values
        
        for col in preheader_features_renamed.columns:
            features[col] = preheader_features_renamed[col].values
        
        # Add categorical features
        for df_cat in [dialog_dummies, syfte_dummies, product_dummies]:
            for col in df_cat.columns:
                features[col] = df_cat[col].values
        
        # Add age features
        features['Min_age'] = campaign_metadata.get('min_age', 18)
        features['Max_age'] = campaign_metadata.get('max_age', 100)
        features['Mean_age'] = (features['Min_age'] + features['Max_age']) / 2
        
        # Add relationship between subject and preheader
        features['Subject_preheader_length_ratio'] = (
            len(subject) / max(len(preheader), 1)
        )
        
        # Calculate text similarity between subject and preheader
        features['Subject_preheader_word_overlap'] = calculate_word_overlap(
            pd.Series([subject]), pd.Series([preheader])
        )[0]
        
        # Create specialized feature sets for each KPI
        features_dict = {}
        
        # Openrate features
        openrate_features = features.copy()
        openrate_features['Subject_first_word_is_question'] = 1 if subject.split() and subject.split()[0].endswith('?') else 0
        openrate_features['Subject_contains_number'] = 1 if any(c.isdigit() for c in subject) else 0
        openrate_features['Subject_contains_personalization'] = 1 if '{{' in subject else 0
        openrate_features['Preheader_complements_subject'] = 1 if preheader and preheader[:20] not in subject and len(preheader) > 10 else 0
        
        # Clickrate features
        clickrate_features = features.copy()
        # For a new campaign, we don't know the open rate yet
        # We can use a placeholder or a predicted value
        clickrate_features['Openrate'] = 0.2  # Placeholder
        clickrate_features['Subject_has_cta'] = 1 if any(cta in subject.lower() for cta in ['klicka', 'läs mer', 'se här', 'upptäck']) else 0
        clickrate_features['Preheader_has_cta'] = 1 if any(cta in preheader.lower() for cta in ['klicka', 'läs mer', 'se här', 'upptäck']) else 0
        clickrate_features['Subject_suggests_value'] = 1 if any(val in subject.lower() for val in ['rabatt', 'erbjudande', 'spara', 'ny', 'exklusiv']) else 0
        clickrate_features['Preheader_suggests_value'] = 1 if any(val in preheader.lower() for val in ['rabatt', 'erbjudande', 'spara', 'ny', 'exklusiv']) else 0
        
        # Optoutrate features
        optoutrate_features = features.copy()
        # For a new campaign, we don't know engagement metrics yet
        # We can use placeholders or predicted values
        optoutrate_features['Openrate'] = 0.2  # Placeholder
        optoutrate_features['Clickrate'] = 0.02  # Placeholder
        optoutrate_features['Subject_all_caps'] = 1 if subject.isupper() else 0
        optoutrate_features['Subject_has_urgency'] = 1 if any(urg in subject.lower() for urg in ['nu', 'idag', 'sista chansen', 'missa inte']) else 0
        optoutrate_features['Subject_many_exclamations'] = 1 if subject.count('!') > 1 else 0
        
        # Add to dictionary
        features_dict['openrate'] = openrate_features
        features_dict['clickrate'] = clickrate_features
        features_dict['optoutrate'] = optoutrate_features
        
        return features_dict
        
    except Exception as e:
        logger.error(f"Error creating campaign-level features: {str(e)}")
        logger.error(traceback.format_exc())
        raise