import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
import pickle
import json
import requests
from dotenv import load_dotenv
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables for API key
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Define the enum values
BOLAG_VALUES = {
    "Blekinge": "Blekinge", "Dalarna": "Dalarna", "Älvsborg": "Älvsborg", "Gävleborg": "Gävleborg",
    "Göinge-Kristianstad": "Göinge-Kristianstad", "Göteborg och Bohuslän": "Göteborg och Bohuslän", "Halland": "Halland",
    "Jämtland": "Jämtland", "Jönköping": "Jönköping", "Kalmar": "Kalmar", "Kronoberg": "Kronoberg",
    "Norrbotten": "Norrbotten", "Skaraborg": "Skaraborg", "Stockholm": "Stockholm", "Södermanland": "Södermanland",
    "Uppsala": "Uppsala", "Värmland": "Värmland", "Västerbotten": "Västerbotten", "Västernorrland": "Västernorrland",
    "Bergslagen": "Bergslagen", "Östgöta": "Östgöta", "Gotland": "Gotland", "Skåne": "Skåne"
}

SYFTE_VALUES = {
    "AKUT": ["AKT", "AKUT"], "AVSLUT": ["AVS", "AVSLUT"], "AVSLUT Kund": ["AVS_K", "AVSLUT Kund"],
    "AVSLUT Produkt": ["AVS_P", "AVSLUT Produkt"], "BEHÅLLA": ["BHA", "BEHÅLLA"],
    "BEHÅLLA Betalpåminnelse": ["BHA_P", "BEHÅLLA Betalpåminnelse"], "BEHÅLLA Inför förfall": ["BHA_F", "BEHÅLLA Inför förfall"],
    "TEST": ["TST", "TEST"], "VINNA": ["VIN", "VINNA"], "VINNA Provapå till riktig": ["VIN_P", "VINNA Provapå till riktig"],
    "VÄLKOMNA": ["VLK", "VÄLKOMNA"], "VÄLKOMNA Nykund": ["VLK_K", "VÄLKOMNA Nykund"],
    "VÄLKOMNA Nyprodukt": ["VLK_P", "VÄLKOMNA Nyprodukt"], "VÄLKOMNA Tillbaka": ["VLK_T", "VÄLKOMNA Tillbaka"],
    "VÄXA": ["VXA", "VÄXA"], "VÄXA Korsförsäljning": ["VXA_K", "VÄXA Korsförsäljning"],
    "VÄXA Merförsäljning": ["VXA_M", "VÄXA Merförsäljning"], "VÅRDA": ["VRD", "VÅRDA"],
    "VÅRDA Betalsätt": ["VRD_B", "VÅRDA Betalsätt"], "VÅRDA Event": ["VRD_E", "VÅRDA Event"],
    "VÅRDA Information": ["VRD_I", "VÅRDA Information"], "VÅRDA Lojalitet förmånskund": ["VRD_L", "VÅRDA Lojalitet förmånskund"],
    "VÅRDA Nyhetsbrev": ["VRD_N", "VÅRDA Nyhetsbrev"], "VÅRDA Skadeförebygg": ["VRD_S", "VÅRDA Skadeförebygg"],
    "VÅRDA Undersökning": ["VRD_U", "VÅRDA Undersökning"], "ÅTERTAG": ["ATG", "ÅTERTAG"],
    "ÖVRIGT": ["OVR", "ÖVRIGT"]
}

DIALOG_VALUES = {
    "BANK": ["BNK", "BANK"], "BANK LFF": ["LFF", "BANK LFF"], "BOENDE": ["BO", "BOENDE"],
    "DROP-OFF": ["DRP", "DROP-OFF"], "FORDON": ["FRD", "FORDON"], "FÖRETAGARBREVET": ["FTB", "FÖRETAGARBREVET"],
    "FÖRETAGSFÖRSÄKRING": ["FTG", "FÖRETAGSFÖRSÄKRING"], "HÄLSA": ["H", "HÄLSA"],
    "KUNDNIVÅ Förmånskund": ["KFB", "KUNDNIVÅ Förmånskund"], "LIV": ["LIV", "LIV"],
    "Livshändelse": ["LVS", "Livshändelse"], "MÅN A1 - Barnförsäkring": ["A1", "MÅN A1 - Barnförsäkring"],
    "MÅN A10 - Förra veckans sålda": ["A10", "MÅN A10 - Förra veckans sålda"],
    "MÅN A3 - Återtag boendeförsäkring": ["A3", "MÅN A3 - Återtag boendeförsäkring"],
    "MÅN A7 - Återtag bilförsäkring": ["A7", "MÅN A7 - Återtag bilförsäkring"],
    "MÅN C2 - Boende till bil": ["C2", "MÅN C2 - Boende till bil"],
    "MÅN C3 - Bilförsäkring förfaller hos konkurrent": ["C3", "MÅN C3 - Bilförsäkring förfaller hos konkurrent"],
    "MÅN F10 - Fasträntekontor": ["F10", "MÅN F10 - Fasträntekontor"],
    "MÅN L1 - Bolån till boendeförsäkringskunder": ["L1", "MÅN L1 - Bolån till boendeförsäkringskunder"],
    "MÅN L20 - Förfall bolån": ["L20", "MÅN L20 - Förfall bolån"], "MÅN L3 - Ränteförfall": ["L3", "MÅN L3 - Ränteförfall"],
    "MÅN M1 - Märkespaket": ["M1", "MÅN M1 - Märkespaket"], "MÅN S1 - Vända pengar": ["S1", "MÅN S1 - Vända pengar"],
    "MÅN S2 - Inflytt pensionskapital": ["S2", "MÅN S2 - Inflytt pensionskapital"], "NBO": ["FNO", "NBO"],
    "OFFERT": ["OF", "OFFERT"], "ONEOFF": ["ONE", "ONEOFF"], "PERSON": ["P", "PERSON"],
    "RÄDDA KVAR": ["RKR", "RÄDDA KVAR"], "TESTUTSKICK": ["TST", "TESTUTSKICK"], "ÅTERBÄRING": ["ATB", "ÅTERBÄRING"]
}

PRODUKT_VALUES = {
    "AGRIA": ["A_A_", "AGRIA"], "BANK": ["B_B_", "BANK"], "BANK Bolån": ["B_B_B_", "BANK Bolån"],
    "BANK Kort": ["B_K_", "BANK Kort"], "BANK Spar": ["B_S_", "BANK Spar"], "BANK Övriga lån": ["B_PL_", "BANK Övriga lån"],
    "BO": ["BO_", "BO"], "BO Alarm": ["BO_AL_", "BO Alarm"], "BO BRF": ["BO_BR_", "BO BRF"],
    "BO Fritid": ["BO_F_", "BO Fritid"], "BO HR": ["BO_HR_", "BO HR"], "BO Villa": ["BO_V_", "BO Villa"],
    "BO VillaHem": ["BO_VH_", "BO VillaHem"], "BÅT": ["BT_", "BÅT"], "FOND": ["F_F_", "FOND"],
    "FÖRETAG Företagarförsäkring": ["F_F_F_", "FÖRETAG Företagarförsäkring"],
    "FÖRETAG Företagarförsäkring prova på": ["F_F_PR_", "FÖRETAG Företagarförsäkring prova på"],
    "HÄLSA": ["H_H_", "HÄLSA"], "HÄLSA BoKvar": ["H_B_", "HÄLSA BoKvar"], "HÄLSA Diagnos": ["H_D_", "HÄLSA Diagnos"],
    "HÄLSA Grupp företag": ["H_G_", "HÄLSA Grupp företag"], "HÄLSA Olycksfall": ["H_O_", "HÄLSA Olycksfall"],
    "HÄLSA Sjukersättning": ["H_S_", "HÄLSA Sjukersättning"], "HÄLSA Sjukvårdsförsäkring": ["H_SV_", "HÄLSA Sjukvårdsförsäkring"],
    "INGEN SPECIFIK PRODUKT": ["NA_NA_", "INGEN SPECIFIK PRODUKT"], "LANTBRUK": ["LB_", "LANTBRUK"],
    "LIV": ["L_L_", "LIV"], "LIV Försäkring": ["L_F_", "LIV Försäkring"], "LIV Pension": ["L_P_", "LIV Pension"],
    "MOTOR": ["M_M_", "MOTOR"], "MOTOR Personbil": ["M_PB_", "MOTOR Personbil"],
    "MOTOR Personbil Vagnskada": ["M_PB_VG_", "MOTOR Personbil Vagnskada"],
    "MOTOR Personbil märkes Lexus": ["M_PB_ML_", "MOTOR Personbil märkes Lexus"],
    "MOTOR Personbil märkes Suzuki": ["M_PB_MS_", "MOTOR Personbil märkes Suzuki"],
    "MOTOR Personbil märkes Toyota": ["M_PB_MT_", "MOTOR Personbil märkes Toyota"],
    "MOTOR Personbil prova på": ["M_PB_PR_", "MOTOR Personbil prova på"], "MOTOR Övriga": ["M_OV_", "MOTOR Övriga"],
    "MOTOR Övriga MC": ["M_OV_MC_", "MOTOR Övriga MC"], "MOTOR Övriga Skoter": ["M_OV_SKO_", "MOTOR Övriga Skoter"],
    "MOTOR Övriga Släp": ["M_OV_SLP_", "MOTOR Övriga Släp"], "PERSON": ["P_P_", "PERSON"],
    "PERSON 60plus": ["P_60_", "PERSON 60plus"], "PERSON Gravid": ["P_G_", "PERSON Gravid"],
    "PERSON Gravid bas": ["P_G_B_", "PERSON Gravid bas"], "PERSON Gravid plus": ["P_G_P_", "PERSON Gravid plus"],
    "PERSON OB": ["P_B_", "PERSON OB"], "PERSON OSB": ["P_OSB_", "PERSON OSB"], "PERSON OSV": ["P_OSV_", "PERSON OSV"]
}

# Create reverse mappings for codes to values
DIALOG_CODE_TO_VALUE = {code: key for key, vals in DIALOG_VALUES.items() for code in vals if code != key}
SYFTE_CODE_TO_VALUE = {code: key for key, vals in SYFTE_VALUES.items() for code in vals if code != key}
PRODUKT_CODE_TO_VALUE = {code: key for key, vals in PRODUKT_VALUES.items() for code in vals if code != key}

# Helper functions
def load_data():
    """Load delivery and customer data from CSV files."""
    try:
        delivery_df = pd.read_csv("./data/delivery_data.csv", encoding='utf-8', sep=';')
        customer_df = pd.read_csv("./data/customer_data.csv", encoding='utf-8', sep=';')
        return delivery_df, customer_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
        
def categorize_age(age):
    """
    Categorize age into predefined age groups
    """
    if 18 <= age <= 24:
        return '18-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 39:
        return '30-39'
    elif 40 <= age <= 49:
        return '40-49'
    elif 50 <= age <= 59:
        return '50-59'
    elif 60 <= age <= 69:
        return '60-69'
    elif 70 <= age <= 79:
        return '70-79'
    elif 80 <= age <= 89:
        return '80-89'
    elif 90 <= age <= 100:
        return '90-100'
    else:
        return 'Other'

def preprocess_data(delivery_df, customer_df):
    """Preprocess and merge the delivery and customer data."""
    if delivery_df is None or customer_df is None:
        return None
    
    # Clean delivery data
    delivery_df = delivery_df.copy()
    
    # Define age bins for consistent use throughout the app
    AGE_BINS = [0, 25, 35, 45, 55, 65, 100]
    AGE_LABELS = ['0-25', '26-35', '36-45', '46-55', '56-65', '65+']
    
    # Convert Date to datetime
    delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
    delivery_df['Year'] = delivery_df['Date'].dt.year
    delivery_df['Month'] = delivery_df['Date'].dt.month
    delivery_df['DayOfWeek'] = delivery_df['Date'].dt.dayofweek
    delivery_df['Hour'] = delivery_df['Date'].dt.hour
    
    # Calculate rates
    delivery_df['OpenRate'] = delivery_df['Opens'] / delivery_df['Sendouts']
    delivery_df['ClickRate'] = delivery_df['Clicks'] / delivery_df['Sendouts']
    delivery_df['OptoutRate'] = delivery_df['Optouts'] / delivery_df['Sendouts']
    
    # Map codes to full values for categorical features
    for col, mapping in zip(['Dialog', 'Syfte', 'Produkt'], 
                           [DIALOG_CODE_TO_VALUE, SYFTE_CODE_TO_VALUE, PRODUKT_CODE_TO_VALUE]):
        if col in delivery_df.columns:
            # First check if the value is already a full value
            for key in mapping.values():
                mask = delivery_df[col] == key
                if mask.any():
                    continue
            
            # If not, try to map from code to full value
            delivery_df[col] = delivery_df[col].apply(
                lambda x: mapping.get(x, x) if isinstance(x, str) else x
            )
    
    # Clean and aggregate customer data
    customer_df = customer_df.copy()
    
    # Ensure all binary variables are numeric
    for col in ['OptOut', 'Open', 'Click']:
        if col in customer_df.columns:
            customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce').fillna(0)
    
    # Map Bolag if necessary
    if 'Bolag' in customer_df.columns:
        for key in BOLAG_VALUES.keys():
            mask = customer_df['Bolag'] == key
            if mask.any():
                continue
        
        # If not found in full values, try to map
        customer_df['Bolag'] = customer_df['Bolag'].apply(
            lambda x: BOLAG_VALUES.get(x, x) if isinstance(x, str) else x
        )
    
    # Aggregate customer data at InternalName level
    customer_agg = customer_df.groupby('InternalName').agg({
        'Primary key': 'count',
        'OptOut': 'mean',
        'Open': 'mean',
        'Click': 'mean',
        'Gender': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None,
        'Age': 'mean',
        'Bolag': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
    }).reset_index()
    
    customer_agg.rename(columns={
        'Primary key': 'UniqueCustomers',
        'OptOut': 'OptOutRateCustomer',
        'Open': 'OpenRateCustomer',
        'Click': 'ClickRateCustomer',
        'Gender': 'MostCommonGender',
        'Age': 'AverageAge',
        'Bolag': 'MostCommonBolag'
    }, inplace=True)
    
    # Merge delivery and aggregated customer data
    merged_df = delivery_df.merge(customer_agg, on='InternalName', how='left')
    
    # Extract text features
    merged_df['SubjectLength'] = merged_df['Subject'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    merged_df['PreheaderLength'] = merged_df['Preheader'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Remove rows with missing target values
    merged_df = merged_df.dropna(subset=['OpenRate', 'ClickRate', 'OptoutRate'])
    
    # Cap extreme values (outliers) for rates
    for col in ['OpenRate', 'ClickRate', 'OptoutRate']:
        upper_limit = merged_df[col].quantile(0.99)
        merged_df[col] = merged_df[col].clip(upper=upper_limit)
    
    return merged_df

def extract_text_features(df, text_col):
    """Extract TF-IDF features from text columns."""
    if df is None or text_col not in df.columns:
        return None, None
    
    # Fill NaN values with empty string
    texts = df[text_col].fillna('').astype(str)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limit features to avoid dimensionality issues
        min_df=3,          # Minimum document frequency
        stop_words=None,   # No stopwords (since we have Swedish text)
        ngram_range=(1, 2) # Include unigrams and bigrams
    )
    
    # Extract features
    features = vectorizer.fit_transform(texts)
    
    return features, vectorizer

def train_model(X, y, model_type='gradient_boosting', sample_weight=None):
    """Train a regression model for predicting rates."""
    if X is None or y is None:
        return None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use sample weights if provided
    if sample_weight is not None:
        train_weight = sample_weight.iloc[X_train.index]
    else:
        train_weight = None
    
    # Choose the model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # Default to Ridge regression
        model = Ridge(alpha=1.0)
    
    # Train the model
    model.fit(X_train, y_train, sample_weight=train_weight)
    
    # Evaluate the model
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    metrics = {
        'train_mse': mean_squared_error(y_train, train_preds),
        'test_mse': mean_squared_error(y_test, test_preds),
        'train_mae': mean_absolute_error(y_train, train_preds),
        'test_mae': mean_absolute_error(y_test, test_preds),
        'train_r2': r2_score(y_train, train_preds),
        'test_r2': r2_score(y_test, test_preds)
    }
    
    return model, metrics, X_test, y_test, test_preds

def cross_validate_model(X, y, model_type='gradient_boosting', n_splits=5, sample_weight=None):
    """Perform cross-validation for model evaluation."""
    if X is None or y is None:
        return None
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_mse = []
    cv_rmse = []
    cv_mae = []
    cv_r2 = []
    
    # Choose the model
    if model_type == 'random_forest':
        model_cls = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model_cls = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # Default to Ridge regression
        model_cls = Ridge(alpha=1.0)
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply sample weights if provided
        if sample_weight is not None:
            train_weight = sample_weight.iloc[train_idx]
        else:
            train_weight = None
            
        # Train the model
        model = model_cls
        model.fit(X_train, y_train, sample_weight=train_weight)
        
        # Evaluate
        y_pred = model.predict(X_val)
        
        cv_mse.append(mean_squared_error(y_val, y_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_mae.append(mean_absolute_error(y_val, y_pred))
        cv_r2.append(r2_score(y_val, y_pred))
    
    cv_results = {
        'mse': {
            'scores': cv_mse,
            'mean': np.mean(cv_mse),
            'std': np.std(cv_mse)
        },
        'rmse': {
            'scores': cv_rmse,
            'mean': np.mean(cv_rmse),
            'std': np.std(cv_rmse)
        },
        'mae': {
            'scores': cv_mae,
            'mean': np.mean(cv_mae),
            'std': np.std(cv_mae)
        },
        'r2': {
            'scores': cv_r2,
            'mean': np.mean(cv_r2),
            'std': np.std(cv_r2)
        }
    }
    
    return cv_results

def prepare_features(df, text_vectorizers=None):
    """Prepare features for model training and prediction."""
    if df is None:
        return None
    
    # Define categorical and numerical features
    categorical_features = ['Dialog', 'Syfte', 'Produkt', 'MostCommonGender', 'MostCommonBolag', 
                           'Year', 'Month', 'DayOfWeek']
    
    # Check if 'AgeBin' column exists, if so use it instead of 'AverageAge'
    if 'AgeBin' in df.columns:
        categorical_features.append('AgeBin')
        numerical_features = ['Hour', 'SubjectLength', 'PreheaderLength', 'UniqueCustomers']
    else:
        numerical_features = ['Hour', 'SubjectLength', 'PreheaderLength', 'AverageAge', 'UniqueCustomers']
    
    # Prepare categorical features
    categorical_df = df[categorical_features].copy()
    for col in categorical_features:
        if col in df.columns:
            categorical_df[col] = categorical_df[col].astype(str).fillna('unknown')
    
    # Prepare numerical features
    numerical_df = df[numerical_features].copy()
    for col in numerical_features:
        if col in df.columns:
            numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce').fillna(0)
    
    # Extract text features if vectorizers are provided
    if text_vectorizers:
        subject_features = text_vectorizers['subject'].transform(df['Subject'].fillna('').astype(str))
        preheader_features = text_vectorizers['preheader'].transform(df['Preheader'].fillna('').astype(str))
        
        # Convert sparse matrices to dense arrays
        subject_df = pd.DataFrame(subject_features.toarray(), 
                                index=df.index, 
                                columns=[f'subject_{i}' for i in range(subject_features.shape[1])])
        preheader_df = pd.DataFrame(preheader_features.toarray(), 
                                   index=df.index, 
                                   columns=[f'preheader_{i}' for i in range(preheader_features.shape[1])])
        
        # Concatenate all features
        features_df = pd.concat([categorical_df, numerical_df, subject_df, preheader_df], axis=1)
    else:
        features_df = pd.concat([categorical_df, numerical_df], axis=1)
    
    return features_df

def save_model(model, filename):
    """Save a trained model to disk."""
    try:
        os.makedirs('models', exist_ok=True)
        with open(f'models/{filename}', 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def load_model(filename):
    """Load a trained model from disk."""
    try:
        with open(f'models/{filename}', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main Streamlit app
def process_data_for_age_heatmap(df, customer_data):
    """
    Process data for age group heatmap visualization
    """
    try:
        # Add age group to customer data
        customer_data = customer_data.copy()
        customer_data['AgeGroup'] = customer_data['Age'].apply(categorize_age)
        
        # Merge customer data with delivery data
        merged_data = df.merge(customer_data, on='InternalName', how='left')
        
        # Ensure we have all age groups (even if zero data)
        all_age_groups = ['18-24', '25-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
        
        # Calculate rates by age group
        results = []
        
        # Create dictionaries to store aggregated results
        agg_by_dialog = {}
        agg_by_syfte = {}
        agg_by_produkt = {}
        
        # Process by age group
        for age_group in all_age_groups:
            age_data = merged_data[merged_data['AgeGroup'] == age_group]
            
            if len(age_data) == 0:
                # Add a row with zeros if no data for this age group
                results.append({
                    'AgeGroup': age_group,
                    'OpenRate': 0,
                    'ClickRate': 0,
                    'OptoutRate': 0,
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
                    'OpenRate': openrate,
                    'ClickRate': clickrate,
                    'OptoutRate': optoutrate,
                    'Count': len(age_data)
                })
                
                # Aggregate by Dialog
                for dialog in age_data['Dialog'].unique():
                    dialog_data = age_data[age_data['Dialog'] == dialog]
                    dialog_sendouts = dialog_data['Sendouts'].sum()
                    if dialog_sendouts > 0:
                        dialog_opens = dialog_data['Opens'].sum()
                        dialog_clicks = dialog_data['Clicks'].sum()
                        dialog_optouts = dialog_data['Optouts'].sum()
                        
                        if dialog not in agg_by_dialog:
                            agg_by_dialog[dialog] = {}
                        
                        agg_by_dialog[dialog][age_group] = {
                            'OpenRate': dialog_opens / dialog_sendouts if dialog_sendouts > 0 else 0,
                            'ClickRate': dialog_clicks / dialog_sendouts if dialog_sendouts > 0 else 0,
                            'OptoutRate': dialog_optouts / dialog_sendouts if dialog_sendouts > 0 else 0,
                            'Count': len(dialog_data)
                        }
                
                # Aggregate by Syfte
                for syfte in age_data['Syfte'].unique():
                    syfte_data = age_data[age_data['Syfte'] == syfte]
                    syfte_sendouts = syfte_data['Sendouts'].sum()
                    if syfte_sendouts > 0:
                        syfte_opens = syfte_data['Opens'].sum()
                        syfte_clicks = syfte_data['Clicks'].sum()
                        syfte_optouts = syfte_data['Optouts'].sum()
                        
                        if syfte not in agg_by_syfte:
                            agg_by_syfte[syfte] = {}
                        
                        agg_by_syfte[syfte][age_group] = {
                            'OpenRate': syfte_opens / syfte_sendouts if syfte_sendouts > 0 else 0,
                            'ClickRate': syfte_clicks / syfte_sendouts if syfte_sendouts > 0 else 0,
                            'OptoutRate': syfte_optouts / syfte_sendouts if syfte_sendouts > 0 else 0,
                            'Count': len(syfte_data)
                        }
                
                # Aggregate by Produkt
                for produkt in age_data['Produkt'].unique():
                    produkt_data = age_data[age_data['Produkt'] == produkt]
                    produkt_sendouts = produkt_data['Sendouts'].sum()
                    if produkt_sendouts > 0:
                        produkt_opens = produkt_data['Opens'].sum()
                        produkt_clicks = produkt_data['Clicks'].sum()
                        produkt_optouts = produkt_data['Optouts'].sum()
                        
                        if produkt not in agg_by_produkt:
                            agg_by_produkt[produkt] = {}
                        
                        agg_by_produkt[produkt][age_group] = {
                            'OpenRate': produkt_opens / produkt_sendouts if produkt_sendouts > 0 else 0,
                            'ClickRate': produkt_clicks / produkt_sendouts if produkt_sendouts > 0 else 0,
                            'OptoutRate': produkt_optouts / produkt_sendouts if produkt_sendouts > 0 else 0,
                            'Count': len(produkt_data)
                        }
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create a DataFrame for each metric
        metrics = ['OpenRate', 'ClickRate', 'OptoutRate']
        heatmap_data = {metric: pd.DataFrame(index=all_age_groups) for metric in metrics}
        
        # Add overall data
        for metric in metrics:
            heatmap_data[metric]['Overall'] = results_df.set_index('AgeGroup')[metric]
        
        # Add dialog, syfte, and produkt data
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
        
        for produkt, data in agg_by_produkt.items():
            for metric in metrics:
                produkt_values = [data.get(age_group, {}).get(metric, 0) for age_group in all_age_groups]
                display_produkt = next((label for code, label in PRODUKT_VALUES.items() if code[0] == produkt), produkt)
                heatmap_data[metric][f"Produkt: {display_produkt}"] = produkt_values
        
        return heatmap_data, results_df
    except Exception as e:
        st.error(f"Error processing data for age heatmap: {str(e)}")
        return None, None

def prepare_version_heatmap_data(all_options, heatmap_data, metric):
    """
    Prepare data for version comparison heatmap with proportional estimates for different versions
    """
    # Get base data and age groups
    base_data = heatmap_data[metric].copy()
    age_groups = base_data.index
    
    # Create new DataFrame with just age groups
    version_data = pd.DataFrame(index=age_groups)
    
    # Get the baseline data and baseline overall rate
    baseline_values = base_data['Overall'].values
    
    # For OpenRate: use the predicted values from the model
    if metric == 'OpenRate':
        # Find the baseline (version A) open rate
        baseline_overall = next(rate for ver, _, _, rate in all_options if ver == 'A')
        
        for version, _, _, predicted_rate in all_options:
            # Calculate ratio between this version's predicted rate and baseline
            if baseline_overall > 0:
                adjustment_ratio = predicted_rate / baseline_overall
            else:
                adjustment_ratio = 1.0
                
            # Apply ratio to adjust each age group's rate
            # Use numpy's clip to ensure values stay in reasonable range (0-100%)
            adjusted_values = np.clip(baseline_values * adjustment_ratio, 0, 1)
            
            # Add to dataframe
            version_data[f"Version {version}"] = adjusted_values
    
    # For ClickRate and OptoutRate: simulate effect based on open rate change
    else:
        # Find baseline values
        baseline_overall = next(rate for ver, _, _, rate in all_options if ver == 'A')
        
        for version, _, _, predicted_rate in all_options:
            # Calculate modification based on open rate change (simplified model)
            if baseline_overall > 0:
                ratio = predicted_rate / baseline_overall
                
                if metric == 'ClickRate':
                    # Click rate increases with open rate but with diminishing returns
                    adjustment_ratio = 1.0 + (ratio - 1.0) * 0.7
                else:  # OptoutRate
                    # Optout rate slightly decreases as open rate increases (inverse relationship)
                    adjustment_ratio = 1.0 - (ratio - 1.0) * 0.3
            else:
                adjustment_ratio = 1.0
                
            # Apply adjustment
            adjusted_values = np.clip(baseline_values * adjustment_ratio, 0, 1)
            
            # Add to dataframe
            version_data[f"Version {version}"] = adjusted_values
    
    return version_data

def create_interactive_heatmap(data, metric, title, is_percentage=True, colorscale='Viridis'):
    """
    Create an interactive heatmap using Plotly
    """
    # Format data for heatmap
    z = data.values
    x = data.columns
    y = data.index
    
    # Format values for hover text
    if is_percentage:
        hover_text = [[f"{z[i][j]:.2%}" for j in range(len(x))] for i in range(len(y))]
    else:
        hover_text = [[f"{z[i][j]:.4f}" for j in range(len(x))] for i in range(len(y))]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False,
        text=hover_text,
        hoverinfo='text+x+y',
        colorscale=colorscale
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Category',
        yaxis_title='Age Group',
        xaxis=dict(
            tickangle=-45,
            side='bottom',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            autorange='reversed',  # Important to make age groups go from youngest to oldest
            tickfont=dict(size=10)
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    return fig

def send_to_groq_api(subject_line, preheader, openrate_A, selected_dialog, selected_syfte, selected_product, min_age, max_age, included_bolag):
    """
    Send data to Groq API for subject line and preheader suggestions
    """
    try:
        if not GROQ_API_KEY:
            return {"error": "Groq API key not found. Please set GROQ_API_KEY in .env file."}
        
        prompt = f"""
        I need to create email subject lines and preheaders for a marketing campaign. 
        
        Current subject line: "{subject_line}"
        Current preheader: "{preheader}"
        Predicted open rate: {openrate_A:.2%}
        
        Campaign details:
        - Dialog: {selected_dialog}
        - Syfte (Purpose): {selected_syfte}
        - Product: {selected_product}
        - Age range: {min_age} to {max_age}
        - Target regions: {', '.join(included_bolag)}
        
        Please generate THREE alternative email subject lines and preheaders in Swedish that could improve the open rate.
        Return your response as a JSON object with a 'suggestions' field containing an array of objects, each with 'subject' and 'preheader' fields, like this:
        {{
            "suggestions": [
                {{"subject": "First alternative subject line", "preheader": "First alternative preheader"}},
                {{"subject": "Second alternative subject line", "preheader": "Second alternative preheader"}},
                {{"subject": "Third alternative subject line", "preheader": "Third alternative preheader"}}
            ]
        }}
        """
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are an expert in email marketing optimization. Your task is to generate compelling email subject lines and preheaders in Swedish that maximize open rates."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            },
            timeout=30  # Add timeout
        )
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_data = response.json()
        content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        
        try:
            suggestions_data = json.loads(content)
            return suggestions_data
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON response: {content}")
            return {"error": "Failed to parse API response", "raw_content": content}
    
    except requests.RequestException as e:
        st.error(f"API request error: {str(e)}")
        return {"error": f"API request error: {str(e)}"}
    except Exception as e:
        st.error(f"Error sending to Groq API: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def main():
    st.title("Email Campaign Performance Predictor")
    
    # Define age groups for consistent use throughout the app
    AGE_GROUPS = ['18-24', '25-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
    # For backward compatibility
    AGE_BINS = [0, 25, 35, 45, 55, 65, 100]
    AGE_LABELS = ['0-25', '26-35', '36-45', '46-55', '56-65', '65+']
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Prediction"])
    
    # Load data
    delivery_df, customer_df = load_data()
    
    if delivery_df is None or customer_df is None:
        st.error("Failed to load data. Please make sure the CSV files are in the data/ directory.")
        return
    
    # Preprocess data
    df = preprocess_data(delivery_df, customer_df)
    
    # Add age bins if not already present
    if df is not None and 'AgeBin' not in df.columns and 'AverageAge' in df.columns:
        df['AgeBin'] = pd.cut(df['AverageAge'], bins=AGE_BINS, labels=AGE_LABELS)
    
    if df is None:
        st.error("Error preprocessing data.")
        return
    
    # Extract text features once for reuse
    subject_features, subject_vectorizer = extract_text_features(df, 'Subject')
    preheader_features, preheader_vectorizer = extract_text_features(df, 'Preheader')
    
    text_vectorizers = {
        'subject': subject_vectorizer,
        'preheader': preheader_vectorizer
    }
    
    # Home page
    if page == "Home":
        st.header("Welcome to the Email Campaign Performance Predictor")
        st.write("""
        This application helps you predict the performance of email campaigns based on historical data. 
        You can explore your data, train predictive models, and make predictions for new campaigns.
        
        ### Available Features:
        
        - **Data Exploration**: Analyze your email campaign data and customer data
        - **Model Training**: Train and evaluate models to predict open rates, click rates, and opt-out rates
        - **Prediction**: Make predictions for new campaigns
        
        ### Getting Started:
        
        1. Make sure your data files (`delivery_data.csv` and `customer_data.csv`) are in the `/data` directory
        2. Use the sidebar navigation to explore your data
        3. Train models to predict campaign performance
        4. Make predictions for new campaigns
        """)
        
        # Display data summary
        st.subheader("Data Summary")
        st.write(f"Delivery data: {delivery_df.shape[0]} rows, {delivery_df.shape[1]} columns")
        st.write(f"Customer data: {customer_df.shape[0]} rows, {customer_df.shape[1]} columns")
        st.write(f"Processed data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display quick stats
        st.subheader("Campaign Performance Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_open_rate = df['OpenRate'].mean() * 100
            st.metric("Average Open Rate", f"{avg_open_rate:.2f}%")
        
        with col2:
            avg_click_rate = df['ClickRate'].mean() * 100
            st.metric("Average Click Rate", f"{avg_click_rate:.2f}%")
        
        with col3:
            avg_optout_rate = df['OptoutRate'].mean() * 100
            st.metric("Average Opt-out Rate", f"{avg_optout_rate:.2f}%")
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Explore specific aspects
        exploration_option = st.selectbox("Explore", 
                                         ["Performance Metrics", "Campaign Types", "Customer Demographics", "Time Analysis"])
        
        if exploration_option == "Performance Metrics":
            st.subheader("Performance Metrics Distribution")
            
            metric = st.selectbox("Select Metric", ["OpenRate", "ClickRate", "OptoutRate"])
            
            # Plot histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[metric] * 100, bins=30, kde=True, ax=ax)
            ax.set_xlabel(f"{metric} (%)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {metric}")
            st.pyplot(fig)
            
            # Display statistics
            st.write(f"Mean: {df[metric].mean() * 100:.2f}%")
            st.write(f"Median: {df[metric].median() * 100:.2f}%")
            st.write(f"Standard Deviation: {df[metric].std() * 100:.2f}%")
            st.write(f"Minimum: {df[metric].min() * 100:.2f}%")
            st.write(f"Maximum: {df[metric].max() * 100:.2f}%")
        
        elif exploration_option == "Campaign Types":
            st.subheader("Campaign Performance by Type")
            
            # Choose category to analyze
            category = st.selectbox("Select Category", ["Dialog", "Syfte", "Produkt"])
            metric = st.selectbox("Select Metric", ["OpenRate", "ClickRate", "OptoutRate"])
            
            # Aggregate data
            agg_data = df.groupby(category)[metric].agg(['mean', 'count']).reset_index()
            agg_data['mean'] = agg_data['mean'] * 100  # Convert to percentage
            agg_data = agg_data.sort_values('mean', ascending=False)
            
            # Filter to show only categories with minimum number of campaigns
            min_campaigns = st.slider("Minimum number of campaigns", 1, 50, 5)
            filtered_data = agg_data[agg_data['count'] >= min_campaigns]
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = sns.barplot(x=category, y='mean', data=filtered_data.head(15), ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_xlabel(category)
            ax.set_ylabel(f"{metric} (%)")
            ax.set_title(f"Average {metric} by {category} (min {min_campaigns} campaigns)")
            
            # Add count labels
            for i, p in enumerate(bars.patches):
                if i < len(filtered_data):
                    count = filtered_data.iloc[i]['count']
                    bars.annotate(f"n={count}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='bottom', xytext=(0, 5), 
                                 textcoords='offset points')
            
            st.pyplot(fig)
        
        elif exploration_option == "Customer Demographics":
            st.subheader("Performance by Customer Demographics")
            
            demo_option = st.selectbox("Select Demographic", ["AverageAge", "MostCommonGender", "MostCommonBolag"])
            metric = st.selectbox("Select Metric", ["OpenRate", "ClickRate", "OptoutRate"])
            
            if demo_option == "AverageAge":
                # Create age bins using the predefined bins
                df['AgeBin'] = pd.cut(df['AverageAge'], bins=AGE_BINS, labels=AGE_LABELS)
                
                # Aggregate data
                agg_data = df.groupby('AgeBin')[metric].mean().reset_index()
                agg_data[metric] = agg_data[metric] * 100  # Convert to percentage
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='AgeBin', y=metric, data=agg_data, ax=ax)
                ax.set_xlabel("Age Group")
                ax.set_ylabel(f"{metric} (%)")
                ax.set_title(f"{metric} by Age Group")
                st.pyplot(fig)
            else:
                # Aggregate data
                agg_data = df.groupby(demo_option)[metric].agg(['mean', 'count']).reset_index()
                agg_data['mean'] = agg_data['mean'] * 100  # Convert to percentage
                agg_data = agg_data.sort_values('mean', ascending=False)
                
                # Filter to show only categories with minimum number of campaigns
                min_campaigns = st.slider("Minimum number of campaigns", 1, 50, 5)
                filtered_data = agg_data[agg_data['count'] >= min_campaigns]
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = sns.barplot(x=demo_option, y='mean', data=filtered_data, ax=ax)
                if demo_option == "MostCommonBolag":
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel(demo_option)
                ax.set_ylabel(f"{metric} (%)")
                ax.set_title(f"Average {metric} by {demo_option} (min {min_campaigns} campaigns)")
                
                # Add count labels
                for i, p in enumerate(bars.patches):
                    if i < len(filtered_data):
                        count = filtered_data.iloc[i]['count']
                        bars.annotate(f"n={count}", 
                                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                                     ha='center', va='bottom', xytext=(0, 5), 
                                     textcoords='offset points')
                
                st.pyplot(fig)
        
        elif exploration_option == "Time Analysis":
            st.subheader("Performance by Time")
            
            time_option = st.selectbox("Select Time Dimension", ["Hour", "DayOfWeek", "Month", "Year"])
            metric = st.selectbox("Select Metric", ["OpenRate", "ClickRate", "OptoutRate"])
            
            # Map day of week numbers to names if selected
            if time_option == "DayOfWeek":
                day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                df['DayName'] = df['DayOfWeek'].map(day_names)
                agg_data = df.groupby('DayName')[metric].mean().reset_index()
                agg_data[metric] = agg_data[metric] * 100  # Convert to percentage
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                agg_data['DayName'] = pd.Categorical(agg_data['DayName'], categories=day_order, ordered=True)
                agg_data = agg_data.sort_values('DayName')
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='DayName', y=metric, data=agg_data, ax=ax)
                ax.set_xlabel("Day of Week")
                ax.set_ylabel(f"{metric} (%)")
                ax.set_title(f"{metric} by Day of Week")
                st.pyplot(fig)
            else:
                # Aggregate data
                agg_data = df.groupby(time_option)[metric].mean().reset_index()
                agg_data[metric] = agg_data[metric] * 100  # Convert to percentage
                
                # Sort by time
                agg_data = agg_data.sort_values(time_option)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                if time_option == "Hour":
                    # Line plot for hour
                    sns.lineplot(x=time_option, y=metric, data=agg_data, ax=ax, marker='o')
                else:
                    sns.barplot(x=time_option, y=metric, data=agg_data, ax=ax)
                ax.set_xlabel(time_option)
                ax.set_ylabel(f"{metric} (%)")
                ax.set_title(f"{metric} by {time_option}")
                st.pyplot(fig)
    
    # Model Training page
    elif page == "Model Training":
        st.header("Model Training")
        
        # Advanced model training options
        with st.expander("Advanced Training Options", expanded=True):
            # Option to train models for all KPIs together
            train_all = st.checkbox("Train models for all KPIs (Open, Click, and Opt-out rates)", value=True)
            use_sample_weights = st.checkbox("Use sample weights for training", value=True)
            
            if use_sample_weights:
                st.info("Sample weights will prioritize examples with higher rates during training.")
                
                col1, col2 = st.columns(2)
                with col1:
                    weight_threshold = st.slider("Weight threshold", 0.0, 1.0, 0.5, 0.05,
                                               help="Samples above this threshold get higher weight")
                with col2:
                    weight_ratio = st.slider("Weight ratio", 1.0, 5.0, 2.0, 0.1,
                                           help="Ratio of weights for high vs low samples")
        
        if not train_all:
            # Select a single target variable
            target = st.selectbox("Select Prediction Target", ["OpenRate", "ClickRate", "OptoutRate"])
            targets = [target]
        else:
            targets = ["OpenRate", "ClickRate", "OptoutRate"]
        
        # Prepare features
        features_df = prepare_features(df, text_vectorizers)
        
        if features_df is None:
            st.error("Error preparing features for model training.")
            return
        
        # Select model type
        model_type = st.selectbox("Select Model Type", 
                                 ["gradient_boosting", "random_forest", "ridge"])
        
        # Train models button
        if st.button("Train Models"):
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Train models for each target
            for i, target in enumerate(targets):
                with st.spinner(f"Training model for {target}..."):
                    # Create sample weights if requested
                    if use_sample_weights:
                        # Create weights that prioritize higher values
                        sample_weights = pd.Series(
                            np.where(df[target] > weight_threshold, weight_ratio, 1.0),
                            index=df.index
                        )
                    else:
                        sample_weights = None
                    
                    # Train model
                    model, metrics, X_test, y_test, test_preds = train_model(
                        features_df, df[target], model_type=model_type, sample_weight=sample_weights
                    )
                    
                    # Cross-validation for more robust evaluation
                    cv_results = cross_validate_model(
                        features_df, df[target], model_type=model_type, sample_weight=sample_weights
                    )
                    
                    # Save model
                    model_filename = f"{target.lower()}_{model_type}_model.pkl"
                    vectorizers_filename = "text_vectorizers.pkl"
                    
                    if save_model(model, model_filename) and save_model(text_vectorizers, vectorizers_filename):
                        st.success(f"Model trained and saved as {model_filename}")
                    
                    # Display metrics
                    st.subheader(f"{target} Model Performance Metrics")
                    
                    # First display cross-validation results
                    st.write("Cross-Validation Results (5-fold):")
                    cv_col1, cv_col2, cv_col3 = st.columns(3)
                    cv_col1.metric("Mean R² Score", f"{cv_results['r2']['mean']:.4f} ± {cv_results['r2']['std']:.4f}")
                    cv_col2.metric("Mean RMSE", f"{cv_results['rmse']['mean']:.6f} ± {cv_results['rmse']['std']:.6f}")
                    cv_col3.metric("Mean MAE", f"{cv_results['mae']['mean']:.6f} ± {cv_results['mae']['std']:.6f}")
                    
                    # Then display train/test metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Training Metrics:")
                        st.write(f"Mean Squared Error: {metrics['train_mse']:.6f}")
                        st.write(f"Mean Absolute Error: {metrics['train_mae']:.6f}")
                        st.write(f"R² Score: {metrics['train_r2']:.4f}")
                    
                    with col2:
                        st.write("Testing Metrics:")
                        st.write(f"Mean Squared Error: {metrics['test_mse']:.6f}")
                        st.write(f"Mean Absolute Error: {metrics['test_mae']:.6f}")
                        st.write(f"R² Score: {metrics['test_r2']:.4f}")
                    
                    # Plot actual vs predicted
                    st.subheader("Actual vs Predicted Values")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, test_preds, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"Actual vs Predicted {target}")
                    st.pyplot(fig)
                    
                    # Feature importance (if applicable)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        
                        # Get feature importance
                        importance = model.feature_importances_
                        feature_names = features_df.columns
                        
                        # Sort features by importance
                        indices = np.argsort(importance)[-15:]  # Top 15 features
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plt.barh(range(len(indices)), importance[indices], align='center')
                        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                        plt.xlabel('Feature Importance')
                        plt.title(f'Top 15 Most Important Features for {target}')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Update progress
                progress_bar.progress((i + 1) / len(targets))
            
            # Final success message
            st.success(f"Successfully trained {len(targets)} models!")
    
    # Prediction page
    elif page == "Prediction":
        st.header("Predict Campaign Performance")
        
        # Check if models exist
        # First try to determine which model type to use
        model_types = ['gradient_boosting', 'random_forest', 'ridge']
        found_model_type = None
        
        for model_type in model_types:
            model_files = [
                f'openrate_{model_type}_model.pkl', 
                f'clickrate_{model_type}_model.pkl', 
                f'optoutrate_{model_type}_model.pkl',
                'text_vectorizers.pkl'
            ]
            
            if all(os.path.exists(f'models/{f}') for f in model_files):
                found_model_type = model_type
                break
        
        if not found_model_type:
            st.warning("Models not found. Please train models on the 'Model Training' page first.")
            return
        
        # Load models
        st.info(f"Using {found_model_type} models")
        open_model = load_model(f'openrate_{found_model_type}_model.pkl')
        click_model = load_model(f'clickrate_{found_model_type}_model.pkl')
        optout_model = load_model(f'optoutrate_{found_model_type}_model.pkl')
        text_vectorizers = load_model('text_vectorizers.pkl')
        
        if not all([open_model, click_model, optout_model, text_vectorizers]):
            st.error("Error loading models. Please retrain models.")
            return
        
        # Create a form for input
        st.subheader("Enter Campaign Details")
        
        with st.form("prediction_form"):
            # Email content
            col1, col2 = st.columns(2)
            
            with col1:
                subject = st.text_input("Email Subject", "")
            
            with col2:
                preheader = st.text_input("Email Preheader", "")
            
            # Campaign metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dialog = st.selectbox("Dialog", list(DIALOG_VALUES.keys()))
            
            with col2:
                syfte = st.selectbox("Syfte", list(SYFTE_VALUES.keys()))
            
            with col3:
                produkt = st.selectbox("Produkt", list(PRODUKT_VALUES.keys()))
            
            # Time details
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
            
            with col2:
                month = st.number_input("Month", min_value=1, max_value=12, value=1)
            
            with col3:
                day_of_week = st.selectbox("Day of Week", 
                                         ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                # Convert to number
                day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                          "Friday": 4, "Saturday": 5, "Sunday": 6}
                day_of_week_num = day_map[day_of_week]
            
            with col4:
                hour = st.number_input("Hour (24h)", min_value=0, max_value=23, value=9)
            
            # Customer demographics
            col1, col2, col3 = st.columns(3)
            st.write("Customer Demographics:")
            
            with col1:
                gender = st.selectbox("Most Common Gender", ["Male", "Female", "Unknown"])
            
            with col2:
                age_range = st.selectbox("Age Range", ['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
                # Map the age range to a representative value for the model
                age_map = {'0-25': 22, '26-35': 30, '36-45': 40, '46-55': 50, '56-65': 60, '65+': 70}
                age = age_map[age_range]
            
            with col3:
                bolag = st.selectbox("Most Common Bolag", list(BOLAG_VALUES.keys()))
            
            # Number of unique customers
            unique_customers = st.number_input("Unique Customers", min_value=1, value=5000)
            
            # Subject line and Preheader input with GenAI checkbox
            col1, col2 = st.columns([3, 1])
            with col1:
                subject = st.text_input('Subject Line')
                preheader = st.text_input('Preheader')
            with col2:
                use_genai = st.checkbox('Use GenAI for A/B/C/D Test', value=True, 
                                      help="Use Groq API to generate alternative subject lines and preheaders")
                
            # Submit button
            submitted = st.form_submit_button("Predict Performance")
        
        if submitted:
            # Create a dataframe for prediction
            pred_data = pd.DataFrame({
                'Subject': [subject],
                'Preheader': [preheader],
                'Dialog': [dialog],
                'Syfte': [syfte],
                'Produkt': [produkt],
                'Year': [year],
                'Month': [month],
                'DayOfWeek': [day_of_week_num],
                'Hour': [hour],
                'MostCommonGender': [gender],
                'AverageAge': [age],
                'AgeBin': [age_range],  # Add the age range
                'MostCommonBolag': [bolag],
                'UniqueCustomers': [unique_customers],
                'SubjectLength': [len(subject)],
                'PreheaderLength': [len(preheader)]
            })
            
            # Prepare features
            pred_features = prepare_features(pred_data, text_vectorizers)
            
            if pred_features is None:
                st.error("Error preparing features for prediction.")
                return
            
            # Make predictions
            open_rate_pred = open_model.predict(pred_features)[0]
            click_rate_pred = click_model.predict(pred_features)[0]
            optout_rate_pred = optout_model.predict(pred_features)[0]
            
            # Display predictions
            st.subheader("Predicted Campaign Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Open Rate", f"{open_rate_pred * 100:.2f}%")
                avg_open = df['OpenRate'].mean() * 100
                diff = (open_rate_pred * 100) - avg_open
                st.write(f"Average: {avg_open:.2f}% (Diff: {diff:+.2f}%)")
            
            with col2:
                st.metric("Click Rate", f"{click_rate_pred * 100:.2f}%")
                avg_click = df['ClickRate'].mean() * 100
                diff = (click_rate_pred * 100) - avg_click
                st.write(f"Average: {avg_click:.2f}% (Diff: {diff:+.2f}%)")
            
            with col3:
                st.metric("Opt-out Rate", f"{optout_rate_pred * 100:.2f}%")
                avg_optout = df['OptoutRate'].mean() * 100
                diff = (optout_rate_pred * 100) - avg_optout
                st.write(f"Average: {avg_optout:.2f}% (Diff: {diff:+.2f}%)")
            
            # Additional metrics
            expected_opens = int(open_rate_pred * unique_customers)
            expected_clicks = int(click_rate_pred * unique_customers)
            expected_optouts = int(optout_rate_pred * unique_customers)
            
            st.subheader("Expected Outcomes")
            st.write(f"Sendouts: {unique_customers}")
            st.write(f"Expected Opens: {expected_opens}")
            st.write(f"Expected Clicks: {expected_clicks}")
            st.write(f"Expected Opt-outs: {expected_optouts}")
            
            # Age group analysis
            try:
                with st.expander("Age Group Analysis", expanded=False):
                    # Process data for heatmap
                    heatmap_data, _ = process_data_for_age_heatmap(df, customer_df)
                    
                    # Create tabs for different metrics
                    metric_tabs = st.tabs(["Open Rate", "Click Rate", "Opt-out Rate"])
                    
                    with metric_tabs[0]:
                        open_data = heatmap_data['OpenRate'][['Overall']]
                        fig = create_interactive_heatmap(open_data, 'OpenRate', 'Open Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_tabs[1]:
                        click_data = heatmap_data['ClickRate'][['Overall']]
                        fig = create_interactive_heatmap(click_data, 'ClickRate', 'Click Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_tabs[2]:
                        optout_data = heatmap_data['OptoutRate'][['Overall']]
                        fig = create_interactive_heatmap(optout_data, 'OptoutRate', 'Opt-out Rate by Age Group')
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in age group analysis: {str(e)}")
            
            # A/B/C/D Testing with Groq API
            if use_genai:
                st.subheader("A/B/C/D Testing")
                
                if st.button('Generate Alternative Subject Lines and Preheaders'):
                    with st.spinner("Generating alternatives with Groq API..."):
                        # Send request to Groq API
                        response_data = send_to_groq_api(
                            subject, preheader, 
                            open_rate_pred, 
                            dialog, syfte, produkt,
                            min_age, max_age, 
                            included_bolag
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
                                    alt_subject = sug.get('subject', '')
                                    alt_preheader = sug.get('preheader', '')
                                    
                                    if alt_subject:
                                        # Create prediction data for this alternative
                                        alt_pred_data = pred_data.copy()
                                        alt_pred_data['Subject'] = alt_subject
                                        alt_pred_data['SubjectLength'] = len(alt_subject)
                                        
                                        if alt_preheader:
                                            alt_pred_data['Preheader'] = alt_preheader
                                            alt_pred_data['PreheaderLength'] = len(alt_preheader)
                                        
                                        # Prepare features and predict
                                        alt_pred_features = prepare_features(alt_pred_data, text_vectorizers)
                                        alt_open_rate = open_model.predict(alt_pred_features)[0]
                                        alt_click_rate = click_model.predict(alt_pred_features)[0]
                                        alt_optout_rate = optout_model.predict(alt_pred_features)[0]
                                        
                                        options.append((
                                            chr(65 + i),  # A, B, C, D
                                            alt_subject, 
                                            alt_preheader, 
                                            alt_open_rate,
                                            alt_click_rate, 
                                            alt_optout_rate
                                        ))
                                
                                # Add current option as Version A
                                all_options = [
                                    ('A', subject, preheader, open_rate_pred, click_rate_pred, optout_rate_pred)
                                ] + options
                                
                                # Display all versions in a comparison table
                                st.subheader("A/B/C/D Test Results")
                                
                                # Create an expander for each version
                                for opt, subj, preh, open_rate, click_rate, optout_rate in all_options:
                                    with st.expander(f"Version {opt}", expanded=True):
                                        is_current = opt == 'A'
                                        is_best_open = open_rate == max(o[3] for o in all_options)
                                        
                                        # Add badges
                                        badges = []
                                        if is_current:
                                            badges.append("🔹 Current")
                                        if is_best_open:
                                            badges.append("⭐ Best Open Rate")
                                        
                                        if badges:
                                            st.markdown(f"<div style='margin-bottom:10px'>{' | '.join(badges)}</div>", unsafe_allow_html=True)
                                        
                                        # Display subject and preheader
                                        st.markdown("**Subject:**")
                                        st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px'>{subj}</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("**Preheader:**")
                                        st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px'>{preh}</div>", unsafe_allow_html=True)
                                        
                                        # Display metrics
                                        st.markdown("**Predicted Results:**")
                                        cols = st.columns(3)
                                        
                                        # Calculate differences from baseline (Version A)
                                        base_open = all_options[0][3]
                                        base_click = all_options[0][4]
                                        base_optout = all_options[0][5]
                                        
                                        delta_open = None if opt == 'A' else (open_rate - base_open) * 100
                                        delta_click = None if opt == 'A' else (click_rate - base_click) * 100
                                        delta_optout = None if opt == 'A' else (optout_rate - base_optout) * 100
                                        
                                        cols[0].metric("Open Rate", f"{open_rate * 100:.2f}%", 
                                                      f"{delta_open:+.2f}%" if delta_open is not None else None)
                                        
                                        cols[1].metric("Click Rate", f"{click_rate * 100:.2f}%", 
                                                      f"{delta_click:+.2f}%" if delta_click is not None else None)
                                        
                                        cols[2].metric("Opt-out Rate", f"{optout_rate * 100:.2f}%", 
                                                      f"{delta_optout:+.2f}%" if delta_optout is not None else None)
                                
                                # Version comparison heatmaps
                                st.subheader("Age Group Analysis Across Versions")
                                try:
                                    # Create tabs for different metrics
                                    metric_tabs = st.tabs(["Open Rate", "Click Rate", "Opt-out Rate"])
                                    
                                    with metric_tabs[0]:
                                        # Prepare data with all versions
                                        version_open_data = prepare_version_heatmap_data(
                                            [(v[0], v[1], v[2], v[3]) for v in all_options], 
                                            heatmap_data, 'OpenRate'
                                        )
                                        
                                        # Calculate best version for each age group
                                        best_version_by_age = version_open_data.idxmax(axis=1)
                                        
                                        # Display heatmap
                                        fig = create_interactive_heatmap(version_open_data, 'OpenRate', 
                                                                      'Open Rate by Age Group and Version',
                                                                      colorscale='Viridis')
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show which version is best for each age group
                                        st.subheader("Best Version by Age Group (Open Rate)")
                                        best_df = pd.DataFrame({
                                            'Age Group': best_version_by_age.index,
                                            'Best Version': best_version_by_age.values,
                                            'Estimated Open Rate': [version_open_data.loc[age, ver] for age, ver in 
                                                               zip(best_version_by_age.index, best_version_by_age.values)]
                                        })
                                        st.dataframe(best_df.set_index('Age Group'), use_container_width=True)
                                    
                                    with metric_tabs[1]:
                                        version_click_data = prepare_version_heatmap_data(
                                            [(v[0], v[1], v[2], v[3]) for v in all_options], 
                                            heatmap_data, 'ClickRate'
                                        )
                                        fig = create_interactive_heatmap(version_click_data, 'ClickRate', 
                                                                      'Click Rate by Age Group and Version',
                                                                      colorscale='Blues')
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with metric_tabs[2]:
                                        version_optout_data = prepare_version_heatmap_data(
                                            [(v[0], v[1], v[2], v[3]) for v in all_options], 
                                            heatmap_data, 'OptoutRate'
                                        )
                                        fig = create_interactive_heatmap(version_optout_data, 'OptoutRate', 
                                                                      'Opt-out Rate by Age Group and Version',
                                                                      colorscale='Reds')
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.caption("""Note: These heatmaps show estimated performance by age group for each version.
                                    The estimates are based on the overall predicted open rate and how it might affect different age groups proportionally.""")
                                except Exception as e:
                                    st.error(f"Error displaying age group heatmaps: {str(e)}")
                                
                                # Find the best option
                                best_option = max(all_options, key=lambda x: x[3])
                                
                                # Summary section
                                st.subheader("Summary")
                                st.write(f"Best performing version: **Version {best_option[0]}** with {best_option[3] * 100:.2f}% predicted open rate")
                                
                                if best_option[0] != 'A':
                                    improvement = best_option[3] - open_rate_pred
                                    st.write(f"Improvement over current version: **{improvement * 100:.2f}%**")
                                    st.write(f"Expected additional opens: **{int(improvement * unique_customers)}**")
                            except Exception as e:
                                st.error(f"Error processing alternatives: {str(e)}")
            
            # Provide suggestions for improvement
            st.subheader("Suggestions for Improvement")
            
            suggestions = []
            
            # Check if open rate is below average
            if open_rate_pred < df['OpenRate'].mean():
                suggestions.append("Consider improving the subject line to increase open rates. Subject lines with 40-50 characters often perform better.")
            
            # Check if click rate is below average
            if click_rate_pred < df['ClickRate'].mean():
                suggestions.append("To improve click rates, ensure your email content is relevant and includes clear call-to-action buttons.")
            
            # Check if opt-out rate is above average
            if optout_rate_pred > df['OptoutRate'].mean():
                suggestions.append("High opt-out rates may indicate that the content isn't relevant to recipients. Consider segmenting your audience more carefully.")
            
            # Day of week recommendations
            best_days = df.groupby('DayOfWeek')['OpenRate'].mean().sort_values(ascending=False).index[:3]
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            best_day_names = [day_names[day] for day in best_days]
            
            if day_of_week_num not in best_days:
                suggestions.append(f"Consider sending on {', '.join(best_day_names[:2])} or {best_day_names[2]}, which historically have better performance.")
            
            # Display suggestions
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"{i}. {suggestion}")
            else:
                st.write("Your campaign parameters look good! No specific improvements needed.")

if __name__ == "__main__":
    main()