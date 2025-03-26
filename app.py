import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os
import requests
import json
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import io

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PredictKPI")

# Constants for dropdown values
BOLAG_VALUES = {
    "Blekinge": "Blekinge", "Dalarna": "Dalarna", "Älvsborg": "Älvsborg", "Gävleborg": "Gävleborg",
    "Göinge-Kristianstad": "Göinge-Kristianstad", "Göteborg och Bohuslän": "Göteborg och Bohuslän", "Halland": "Halland",
    "Jämtland": "Jämtland", "Jönköping": "Jönköping", "Kalmar": "Kalmar", "Kronoberg": "Kronoberg",
    "Norrbotten": "Norrbotten", "Skaraborg": "Skaraborg", "Stockholm": "Stockholm", "Södermanland": "Södermanland",
    "Uppsala": "Uppsala", "Värmland": "Värmland", "Västerbotten": "Västerbotten", "Västernorrland": "Västernorrland",
    "Bergslagen": "Bergslagen", "Östgöta": "Östgöta", "Gotland": "Gotland", "Skåne": "Skåne"
}

SYFTE_VALUES = {
    "AKUT": ("AKT", "AKUT"), "AVSLUT": ("AVS", "AVSLUT"), "AVSLUT Kund": ("AVS_K", "AVSLUT Kund"),
    "AVSLUT Produkt": ("AVS_P", "AVSLUT Produkt"), "BEHÅLLA": ("BHA", "BEHÅLLA"),
    "BEHÅLLA Betalpåminnelse": ("BHA_P", "BEHÅLLA Betalpåminnelse"), "BEHÅLLA Inför förfall": ("BHA_F", "BEHÅLLA Inför förfall"),
    "TEST": ("TST", "TEST"), "VINNA": ("VIN", "VINNA"), "VINNA Provapå till riktig": ("VIN_P", "VINNA Provapå till riktig"),
    "VÄLKOMNA": ("VLK", "VÄLKOMNA"), "VÄLKOMNA Nykund": ("VLK_K", "VÄLKOMNA Nykund"),
    "VÄLKOMNA Nyprodukt": ("VLK_P", "VÄLKOMNA Nyprodukt"), "VÄLKOMNA Tillbaka": ("VLK_T", "VÄLKOMNA Tillbaka"),
    "VÄXA": ("VXA", "VÄXA"), "VÄXA Korsförsäljning": ("VXA_K", "VÄXA Korsförsäljning"),
    "VÄXA Merförsäljning": ("VXA_M", "VÄXA Merförsäljning"), "VÅRDA": ("VRD", "VÅRDA"),
    "VÅRDA Betalsätt": ("VRD_B", "VÅRDA Betalsätt"), "VÅRDA Event": ("VRD_E", "VÅRDA Event"),
    "VÅRDA Information": ("VRD_I", "VÅRDA Information"), "VÅRDA Lojalitet förmånskund": ("VRD_L", "VÅRDA Lojalitet förmånskund"),
    "VÅRDA Nyhetsbrev": ("VRD_N", "VÅRDA Nyhetsbrev"), "VÅRDA Skadeförebygg": ("VRD_S", "VÅRDA Skadeförebygg"),
    "VÅRDA Undersökning": ("VRD_U", "VÅRDA Undersökning"), "ÅTERTAG": ("ATG", "ÅTERTAG"),
    "ÖVRIGT": ("OVR", "ÖVRIGT")
}

DIALOG_VALUES = {
    "BANK": ("BNK", "BANK"), "BANK LFF": ("LFF", "BANK LFF"), "BOENDE": ("BO", "BOENDE"),
    "DROP-OFF": ("DRP", "DROP-OFF"), "FORDON": ("FRD", "FORDON"), "FÖRETAGARBREVET": ("FTB", "FÖRETAGARBREVET"),
    "FÖRETAGSFÖRSÄKRING": ("FTG", "FÖRETAGSFÖRSÄKRING"), "HÄLSA": ("H", "HÄLSA"),
    "KUNDNIVÅ Förmånskund": ("KFB", "KUNDNIVÅ Förmånskund"), "LIV": ("LIV", "LIV"),
    "Livshändelse": ("LVS", "Livshändelse"), "MÅN A1 - Barnförsäkring": ("A1", "MÅN A1 - Barnförsäkring"),
    "MÅN A10 - Förra veckans sålda": ("A10", "MÅN A10 - Förra veckans sålda"),
    "MÅN A3 - Återtag boendeförsäkring": ("A3", "MÅN A3 - Återtag boendeförsäkring"),
    "MÅN A7 - Återtag bilförsäkring": ("A7", "MÅN A7 - Återtag bilförsäkring"),
    "MÅN C2 - Boende till bil": ("C2", "MÅN C2 - Boende till bil"),
    "MÅN C3 - Bilförsäkring förfaller hos konkurrent": ("C3", "MÅN C3 - Bilförsäkring förfaller hos konkurrent"),
    "MÅN F10 - Fasträntekontor": ("F10", "MÅN F10 - Fasträntekontor"),
    "MÅN L1 - Bolån till boendeförsäkringskunder": ("L1", "MÅN L1 - Bolån till boendeförsäkringskunder"),
    "MÅN L20 - Förfall bolån": ("L20", "MÅN L20 - Förfall bolån"), "MÅN L3 - Ränteförfall": ("L3", "MÅN L3 - Ränteförfall"),
    "MÅN M1 - Märkespaket": ("M1", "MÅN M1 - Märkespaket"), "MÅN S1 - Vända pengar": ("S1", "MÅN S1 - Vända pengar"),
    "MÅN S2 - Inflytt pensionskapital": ("S2", "MÅN S2 - Inflytt pensionskapital"), "NBO": ("FNO", "NBO"),
    "OFFERT": ("OF", "OFFERT"), "ONEOFF": ("ONE", "ONEOFF"), "PERSON": ("P", "PERSON"),
    "RÄDDA KVAR": ("RKR", "RÄDDA KVAR"), "TESTUTSKICK": ("TST", "TESTUTSKICK"), "ÅTERBÄRING": ("ATB", "ÅTERBÄRING")
}

PRODUKT_VALUES = {
    "AGRIA": ("A_A_", "AGRIA"), "BANK": ("B_B_", "BANK"), "BANK Bolån": ("B_B_B_", "BANK Bolån"),
    "BANK Kort": ("B_K_", "BANK Kort"), "BANK Spar": ("B_S_", "BANK Spar"), "BANK Övriga lån": ("B_PL_", "BANK Övriga lån"),
    "BO": ("BO_", "BO"), "BO Alarm": ("BO_AL_", "BO Alarm"), "BO BRF": ("BO_BR_", "BO BRF"),
    "BO Fritid": ("BO_F_", "BO Fritid"), "BO HR": ("BO_HR_", "BO HR"), "BO Villa": ("BO_V_", "BO Villa"),
    "BO VillaHem": ("BO_VH_", "BO VillaHem"), "BÅT": ("BT_", "BÅT"), "FOND": ("F_F_", "FOND"),
    "FÖRETAG Företagarförsäkring": ("F_F_F_", "FÖRETAG Företagarförsäkring"),
    "FÖRETAG Företagarförsäkring prova på": ("F_F_PR_", "FÖRETAG Företagarförsäkring prova på"),
    "HÄLSA": ("H_H_", "HÄLSA"), "HÄLSA BoKvar": ("H_B_", "HÄLSA BoKvar"), "HÄLSA Diagnos": ("H_D_", "HÄLSA Diagnos"),
    "HÄLSA Grupp företag": ("H_G_", "HÄLSA Grupp företag"), "HÄLSA Olycksfall": ("H_O_", "HÄLSA Olycksfall"),
    "HÄLSA Sjukersättning": ("H_S_", "HÄLSA Sjukersättning"), "HÄLSA Sjukvårdsförsäkring": ("H_SV_", "HÄLSA Sjukvårdsförsäkring"),
    "INGEN SPECIFIK PRODUKT": ("NA_NA_", "INGEN SPECIFIK PRODUKT"), "LANTBRUK": ("LB_", "LANTBRUK"),
    "LIV": ("L_L_", "LIV"), "LIV Försäkring": ("L_F_", "LIV Försäkring"), "LIV Pension": ("L_P_", "LIV Pension"),
    "MOTOR": ("M_M_", "MOTOR"), "MOTOR Personbil": ("M_PB_", "MOTOR Personbil"),
    "MOTOR Personbil Vagnskada": ("M_PB_VG_", "MOTOR Personbil Vagnskada"),
    "MOTOR Personbil märkes Lexus": ("M_PB_ML_", "MOTOR Personbil märkes Lexus"),
    "MOTOR Personbil märkes Suzuki": ("M_PB_MS_", "MOTOR Personbil märkes Suzuki"),
    "MOTOR Personbil märkes Toyota": ("M_PB_MT_", "MOTOR Personbil märkes Toyota"),
    "MOTOR Personbil prova på": ("M_PB_PR_", "MOTOR Personbil prova på"), "MOTOR Övriga": ("M_OV_", "MOTOR Övriga"),
    "MOTOR Övriga MC": ("M_OV_MC_", "MOTOR Övriga MC"), "MOTOR Övriga Skoter": ("M_OV_SKO_", "MOTOR Övriga Skoter"),
    "MOTOR Övriga Släp": ("M_OV_SLP_", "MOTOR Övriga Släp"), "PERSON": ("P_P_", "PERSON"),
    "PERSON 60plus": ("P_60_", "PERSON 60plus"), "PERSON Gravid": ("P_G_", "PERSON Gravid"),
    "PERSON Gravid bas": ("P_G_B_", "PERSON Gravid bas"), "PERSON Gravid plus": ("P_G_P_", "PERSON Gravid plus"),
    "PERSON OB": ("P_B_", "PERSON OB"), "PERSON OSB": ("P_OSB_", "PERSON OSB"), "PERSON OSV": ("P_OSV_", "PERSON OSV")
}

# Create data directories if they don't exist
os.makedirs("Data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Function to load data
def load_data(delivery_file='Data/delivery_data.csv', customer_file='Data/customer_data.csv'):
    """Load delivery and customer data from CSV files"""
    try:
        # Check if files exist
        if not os.path.exists(delivery_file):
            st.error(f"File not found: {delivery_file}")
            st.info("Please upload data files first.")
            return None, None

        if not os.path.exists(customer_file):
            st.error(f"File not found: {customer_file}")
            st.info("Please upload data files first.")
            return None, None
            
        # Load data
        delivery_df = pd.read_csv(delivery_file, sep=';', encoding='utf-8')
        customer_df = pd.read_csv(customer_file, sep=';', encoding='utf-8')
        
        # Log data shape
        logger.info(f"Loaded delivery data: {delivery_df.shape}")
        logger.info(f"Loaded customer data: {customer_df.shape}")
        
        return delivery_df, customer_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None, None

# Function to preprocess data
def preprocess_data(delivery_df, customer_df):
    """Preprocess and merge delivery and customer data"""
    try:
        # Make copies to avoid modifying original dataframes
        delivery = delivery_df.copy()
        customer = customer_df.copy()
        
        # Handle missing columns
        required_delivery_cols = ['InternalName', 'Subject', 'Date', 'Sendouts', 'Opens', 'Clicks', 'Optouts', 'Dialog', 'Syfte', 'Product']
        required_customer_cols = ['Primary key', 'InternalName', 'OptOut', 'Open', 'Click', 'Gender', 'Age', 'Bolag']
        
        # Add missing columns to delivery_df
        for col in required_delivery_cols:
            if col not in delivery.columns:
                if col == 'Date':
                    delivery[col] = pd.Timestamp('now')
                    logger.warning(f"Added missing column {col} with current timestamp")
                elif col == 'Product' and 'Produkt' in delivery.columns:
                    # Handle potential column name mismatch
                    delivery[col] = delivery['Produkt']
                    logger.warning(f"Renamed 'Produkt' column to 'Product'")
                else:
                    delivery[col] = np.nan
                    logger.warning(f"Added missing column {col} with NaN values")
        
        # Add missing columns to customer_df
        for col in required_customer_cols:
            if col not in customer.columns:
                if col in ['OptOut', 'Open', 'Click']:
                    customer[col] = 0
                    logger.warning(f"Added missing column {col} with zeros")
                else:
                    customer[col] = np.nan
                    logger.warning(f"Added missing column {col} with NaN values")
        
        # Convert date column (handle both 'Date' and 'date')
        if 'Date' in delivery.columns:
            delivery['Date'] = pd.to_datetime(delivery['Date'], errors='coerce')
        elif 'date' in delivery.columns:
            delivery['Date'] = pd.to_datetime(delivery['date'], errors='coerce')
            delivery = delivery.drop('date', axis=1)
        else:
            delivery['Date'] = pd.Timestamp('now')
            logger.warning("Date column not found. Using current timestamp.")
        
        # Ensure Preheader column exists
        if 'Preheader' not in delivery.columns:
            delivery['Preheader'] = ''
            logger.warning("Added missing Preheader column with empty strings")
        
        # Calculate KPIs at delivery level
        delivery['Openrate'] = delivery['Opens'] / delivery['Sendouts']
        delivery['Clickrate'] = delivery['Clicks'] / delivery['Sendouts']
        delivery['Optoutrate'] = delivery['Optouts'] / delivery['Sendouts']
        
        # Fix data types
        for col in ['OptOut', 'Open', 'Click']:
            customer[col] = customer[col].astype(int)
        
        customer['Age'] = pd.to_numeric(customer['Age'], errors='coerce')
        
        # Check for age outliers and fix
        customer.loc[customer['Age'] < 18, 'Age'] = 18
        customer.loc[customer['Age'] > 100, 'Age'] = 100
        
        # Group customer data by InternalName to create features
        # Modify the aggregation to avoid using lambda in column name
        customer_features = customer.groupby('InternalName').agg({
            'Age': ['mean', 'min', 'max'],
            'Gender': lambda x: (x == 'Kvinna').mean(),  # Percentage of females
            'Bolag': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Most common Bolag
            'Open': 'mean',  # Average open rate at customer level
            'Click': 'mean',  # Average click rate at customer level
            'OptOut': 'mean'  # Average opt-out rate at customer level
        })
        
        # Flatten multi-level column names and rename lambda column
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        
        # Rename any problematic lambda columns
        if 'Gender_<lambda>' in customer_features.columns:
            customer_features = customer_features.rename(columns={'Gender_<lambda>': 'Gender_pct'})
        
        customer_features = customer_features.reset_index()
        
        # Create Bolag dummy variables
        bolag_dummies = pd.get_dummies(customer['Bolag'], prefix='Bolag')
        
        # Ensure Bolag columns are numeric
        for col in bolag_dummies.columns:
            bolag_dummies[col] = bolag_dummies[col].astype(float)
            
        bolag_agg = customer.join(bolag_dummies).groupby('InternalName')[bolag_dummies.columns].mean()
        bolag_agg = bolag_agg.reset_index()
        
        # Merge all data
        df = delivery.merge(customer_features, on='InternalName', how='left')
        df = df.merge(bolag_agg, on='InternalName', how='left')
        
        # Fill NAs
        df = df.fillna({
            'Age_mean': 40,
            'Age_min': 18,
            'Age_max': 100,
            'Open_mean': 0,
            'Click_mean': 0,
            'OptOut_mean': 0
        })
        
        # Ensure gender percentage column exists with the right name
        if 'Gender_pct' not in df.columns and 'Gender_<lambda>' in df.columns:
            df['Gender_pct'] = df['Gender_<lambda>'].astype(float).fillna(0.5)
        elif 'Gender_pct' not in df.columns:
            df['Gender_pct'] = 0.5
        
        # Add text features
        df = add_text_features(df)
        
        # Ensure all numeric columns are properly typed
        for col in df.columns:
            if col.startswith('Bolag_') or col in ['Age_mean', 'Age_min', 'Age_max', 'Gender_pct']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        st.error(f"Error preprocessing data: {e}")
        raise

def add_text_features(df):
    """Add features derived from text columns"""
    # Subject line features
    df['Subject_length'] = df['Subject'].str.len()
    df['Subject_word_count'] = df['Subject'].str.split().str.len()
    df['Subject_has_question'] = df['Subject'].str.contains('\\?').astype(int)
    df['Subject_has_exclamation'] = df['Subject'].str.contains('!').astype(int)
    df['Subject_has_number'] = df['Subject'].str.contains('\\d').astype(int)
    
    # Calculate caps percentage in subject
    df['Subject_caps_ratio'] = df['Subject'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    # Preheader features if available
    if 'Preheader' in df.columns:
        df['Preheader_length'] = df['Preheader'].str.len()
        df['Preheader_word_count'] = df['Preheader'].str.split().str.len()
        df['Preheader_has_question'] = df['Preheader'].str.contains('\\?').astype(int)
        df['Preheader_has_exclamation'] = df['Preheader'].str.contains('!').astype(int)
        
        # Calculate relationship between subject and preheader
        df['Subject_preheader_ratio'] = df['Subject_length'] / df['Preheader_length'].replace(0, 1)
    
    return df

def train_models(df, force_retrain=False):
    """Train or load XGBoost models for KPI prediction"""
    model_path = "models/kpi_models.pkl"
    
    # If model exists and we're not forcing retrain, load it
    if os.path.exists(model_path) and not force_retrain:
        try:
            models = pd.read_pickle(model_path)
            logger.info("Loaded existing models")
            return models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.warning("Error loading existing models. Training new models.")
    
    # Initialize dictionary to store models
    models = {}
    
    # Features to use for prediction
    features = [
        # Metadata features
        'Dialog', 'Syfte', 'Product',
        
        # Text features
        'Subject_length', 'Subject_word_count', 'Subject_has_question', 
        'Subject_has_exclamation', 'Subject_has_number', 'Subject_caps_ratio',
        
        # Demographic features
        'Age_mean', 'Age_min', 'Age_max'
    ]
    
    # Add Gender percentage column - ensure it's properly named and numeric
    gender_col = [col for col in df.columns if 'Gender' in col]
    if gender_col:
        # Rename the column if it contains '<lambda>'
        if '<lambda>' in gender_col[0]:
            df['Gender_pct'] = df[gender_col[0]].astype(float)
            features.append('Gender_pct')
        else:
            features.append(gender_col[0])
    
    # Add Bolag columns - ensure they are all numeric
    bolag_cols = [col for col in df.columns if col.startswith('Bolag_')]
    for col in bolag_cols:
        # Convert any object columns to numeric
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    features.extend(bolag_cols)
    
    # Add Preheader features if available
    preheader_features = [
        'Preheader_length', 'Preheader_word_count', 'Preheader_has_question',
        'Preheader_has_exclamation', 'Subject_preheader_ratio'
    ]
    
    if all(col in df.columns for col in preheader_features):
        features.extend(preheader_features)
    
    # Categorical columns need special handling
    categorical_features = ['Dialog', 'Syfte', 'Product']
    
    # Create training data with one-hot encoding for categorical features
    X = pd.get_dummies(df[features], columns=categorical_features)
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Clean column names - XGBoost requires clean feature names without [, ] or <
    X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
    
    # Check if we need to save the feature names for later prediction
    models['feature_names'] = X.columns.tolist()
    
    # For each KPI type, train a model
    for kpi in ['Openrate', 'Clickrate', 'Optoutrate']:
        # Define target
        y = df[kpi]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model with appropriate parameters
        if kpi == 'Openrate':
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                enable_categorical=False  # Don't use categorical mode, we've one-hot encoded
            )
        elif kpi == 'Clickrate':
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                enable_categorical=False
            )
        else:  # Optoutrate - rare events need special handling
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                scale_pos_weight=5,  # Helps with imbalanced data
                random_state=42,
                enable_categorical=False
            )
        
        # Train model
        with st.spinner(f"Training {kpi} model..."):
            model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{kpi} model - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Store model and evaluation metrics
        models[kpi] = {
            'model': model,
            'metrics': {
                'rmse': rmse,
                'r2': r2
            }
        }
    
    # Save models to disk
    try:
        pd.to_pickle(models, model_path)
        logger.info(f"Saved models to {model_path}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
    
    return models

def predict_kpis(models, subject, preheader, dialog, syfte, product, age_min, age_max, included_bolag):
    """Predict KPIs for a new campaign"""
    # Check if we have all the necessary models
    if not all(kpi in models for kpi in ['Openrate', 'Clickrate', 'Optoutrate']):
        st.error("Missing one or more KPI models. Please retrain models.")
        return None
    
    # Create a DataFrame with one row containing the input
    input_data = pd.DataFrame({
        'Subject': [subject],
        'Preheader': [preheader],
        'Dialog': [dialog],
        'Syfte': [syfte],
        'Product': [product],
        'Age_min': [age_min],
        'Age_max': [age_max],
        'Age_mean': [(age_min + age_max) / 2],
        'Gender_pct': [0.5]  # Default to 50% female
    })
    
    # Add Bolag columns (set to 1 for included, 0 for excluded)
    for bolag in BOLAG_VALUES.keys():
        bolag_col = f"Bolag_{bolag}"
        input_data[bolag_col] = 1.0 if bolag in included_bolag else 0.0
    
    # Add text features
    input_data = add_text_features(input_data)
    
    # Create dummy variables for categorical features
    categorical_features = ['Dialog', 'Syfte', 'Product']
    X = pd.get_dummies(input_data, columns=categorical_features)
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Clean column names to match training format
    X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
    
    # Get feature names from the model
    feature_names = models['feature_names']
    
    # Add missing columns from training data
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    
    # Keep only the columns used during training and ensure they're in the right order
    X = X[feature_names]
    
    # Make predictions for each KPI
    predictions = {}
    for kpi in ['Openrate', 'Clickrate', 'Optoutrate']:
        model = models[kpi]['model']
        predictions[kpi] = float(model.predict(X)[0])
    
    return predictions

def get_subject_suggestions(subject, preheader, dialog, syfte, product, predictions, age_min, age_max, included_bolag):
    """Get subject line suggestions using Groq API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set it in the .env file.")
        return None
    
    # Format predictions
    openrate = predictions.get('Openrate', 0)
    clickrate = predictions.get('Clickrate', 0)
    optoutrate = predictions.get('Optoutrate', 0)
    
    # Get display names
    dialog_display = DIALOG_VALUES.get(dialog, (dialog, dialog))[1]
    syfte_display = SYFTE_VALUES.get(syfte, (syfte, syfte))[1]
    product_display = PRODUKT_VALUES.get(product, (product, product))[1]
    
    # Create prompt for Groq
    prompt = f"""
    I need to create email subject lines and preheaders for a marketing campaign. 
    
    Current subject line: "{subject}"
    Current preheader: "{preheader}"
    Current predicted metrics:
    - Open rate: {openrate:.2%}
    - Click rate: {clickrate:.2%}
    - Opt-out rate: {optoutrate:.2%}
    
    Campaign details:
    - Dialog: {dialog_display}
    - Syfte (Purpose): {syfte_display}
    - Product: {product_display}
    - Age range: {age_min} to {age_max}
    - Target regions: {', '.join(included_bolag) if included_bolag else 'All regions'}
    
    Please generate THREE alternative email subject lines and preheaders in Swedish that could improve all metrics (higher open and click rates, lower opt-out rate).
    
    Consider these patterns that work well:
    - Clear value proposition
    - Personalization when relevant
    - Avoiding clickbait or urgent language that might increase opt-outs
    - Keeping subject lines between 30-60 characters
    - Making preheaders complementary to subjects, not repetitive
    
    Return your response as a JSON object with a 'suggestions' field containing an array of objects, each with 'subject' and 'preheader' fields, like this:
    {{
        "suggestions": [
            {{"subject": "First alternative subject line", "preheader": "First alternative preheader"}},
            {{"subject": "Second alternative subject line", "preheader": "Second alternative preheader"}},
            {{"subject": "Third alternative subject line", "preheader": "Third alternative preheader"}}
        ]
    }}
    """
    
    try:
        # Send request to Groq API
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are an expert in email marketing optimization."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            },
            verify=False,
            timeout=30
        )
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
        
        # Parse response
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        
        try:
            # Parse JSON content
            suggestions_data = json.loads(content)
            return suggestions_data.get('suggestions', [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq API response: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return None

def compare_suggestions(models, original_subject, original_preheader, suggestions, dialog, syfte, product, age_min, age_max, included_bolag):
    """Compare original subject/preheader with AI suggestions"""
    # Start with the original as option A
    options = [{
        'version': 'A',
        'subject': original_subject,
        'preheader': original_preheader
    }]
    
    # Add suggestions as options B, C, D
    for i, suggestion in enumerate(suggestions[:3]):  # Limit to 3 suggestions
        options.append({
            'version': chr(66 + i),  # B, C, D
            'subject': suggestion.get('subject', ''),
            'preheader': suggestion.get('preheader', '')
        })
    
    # Predict KPIs for each option
    for option in options:
        predictions = predict_kpis(
            models,
            option['subject'],
            option['preheader'],
            dialog,
            syfte,
            product,
            age_min,
            age_max,
            included_bolag
        )
        
        if predictions:
            option['predictions'] = predictions
            
            # Calculate a combined score (higher is better)
            option['score'] = (
                predictions['Openrate'] * 0.5 +
                predictions['Clickrate'] * 0.5 -
                predictions['Optoutrate'] * 2.0
            )
    
    # Sort by score (highest first)
    options.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Get the best option
    best_option = options[0] if options else None
    
    # Calculate improvement if best option is not the original
    improvement = {}
    if best_option and best_option['version'] != 'A':
        original = next((opt for opt in options if opt['version'] == 'A'), None)
        if original:
            improvement = {
                'openrate': best_option['predictions']['Openrate'] - original['predictions']['Openrate'],
                'clickrate': best_option['predictions']['Clickrate'] - original['predictions']['Clickrate'],
                'optoutrate': original['predictions']['Optoutrate'] - best_option['predictions']['Optoutrate']
            }
    
    return {
        'options': options,
        'best_option': best_option,
        'improvement': improvement
    }

def display_comparative_results(results):
    """Display comparative results of original vs AI suggestions"""
    if not results or 'options' not in results:
        st.error("No comparison results to display.")
        return
    
    options = results['options']
    best_option = results.get('best_option')
    improvement = results.get('improvement', {})
    
    # Display best option
    if best_option:
        st.success(f"🏆 Best Option (Version {best_option['version']})")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Subject Line:**")
            st.code(best_option['subject'])
            
            st.markdown("**Preheader:**")
            st.code(best_option['preheader'])
        
        with col2:
            # Show improvement if not the original
            if best_option['version'] != 'A':
                st.metric("Open Rate Improvement", f"{improvement.get('openrate', 0):.2%}")
                st.metric("Click Rate Improvement", f"{improvement.get('clickrate', 0):.2%}")
                st.metric("Opt-out Rate Improvement", f"{improvement.get('optoutrate', 0):.2%}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["All Options", "Comparison"])
    
    with tab1:
        # Display all options
        for option in options:
            with st.expander(f"Version {option['version']}", expanded=(option == best_option)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Subject Line:**")
                    st.code(option['subject'])
                    
                    st.markdown("**Preheader:**")
                    st.code(option['preheader'])
                
                with col2:
                    if 'predictions' in option:
                        st.metric("Open Rate", f"{option['predictions']['Openrate']:.2%}")
                        st.metric("Click Rate", f"{option['predictions']['Clickrate']:.2%}")
                        st.metric("Opt-out Rate", f"{option['predictions']['Optoutrate']:.2%}")
    
    with tab2:
        # Bar chart comparison
        if options:
            # Prepare data
            versions = [o['version'] for o in options]
            openrates = [o['predictions']['Openrate'] for o in options if 'predictions' in o]
            clickrates = [o['predictions']['Clickrate'] for o in options if 'predictions' in o]
            optoutrates = [o['predictions']['Optoutrate'] for o in options if 'predictions' in o]
            
            # Create combined DataFrame for plotting
            plot_data = []
            for i, version in enumerate(versions):
                if i < len(openrates):
                    plot_data.append({'Version': version, 'Metric': 'Open Rate', 'Value': openrates[i]})
                    plot_data.append({'Version': version, 'Metric': 'Click Rate', 'Value': clickrates[i]})
                    plot_data.append({'Version': version, 'Metric': 'Opt-out Rate', 'Value': optoutrates[i]})
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create plot
            fig = px.bar(
                plot_df,
                x='Version',
                y='Value',
                color='Metric',
                barmode='group',
                title='KPI Comparison Across Versions',
                labels={'Value': 'Rate', 'Version': 'Version'},
                text_auto='.2%'
            )
            
            fig.update_layout(yaxis_tickformat='.1%')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create comparison table
            table_data = []
            for option in options:
                if 'predictions' in option:
                    table_data.append({
                        'Version': option['version'],
                        'Subject': option['subject'],
                        'Open Rate': f"{option['predictions']['Openrate']:.2%}",
                        'Click Rate': f"{option['predictions']['Clickrate']:.2%}",
                        'Opt-out Rate': f"{option['predictions']['Optoutrate']:.2%}",
                        'Combined Score': f"{option.get('score', 0):.2f}"
                    })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

def batch_predict(models, subject_lines, preheaders, dialog, syfte, product, age_min, age_max, included_bolag):
    """Perform batch prediction on multiple subject lines"""
    results = []
    
    # Ensure preheaders list matches subject_lines length
    if not preheaders or len(preheaders) != len(subject_lines):
        preheaders = [''] * len(subject_lines)
    
    # Process each subject/preheader pair
    for i, (subject, preheader) in enumerate(zip(subject_lines, preheaders)):
        # Make predictions
        predictions = predict_kpis(
            models,
            subject,
            preheader,
            dialog,
            syfte,
            product,
            age_min,
            age_max,
            included_bolag
        )
        
        if predictions:
            # Calculate a combined score
            combined_score = (
                predictions['Openrate'] * 0.5 +
                predictions['Clickrate'] * 0.5 -
                predictions['Optoutrate'] * 2.0
            )
            
            # Add to results
            results.append({
                'id': i + 1,
                'subject': subject,
                'preheader': preheader,
                'openrate': predictions['Openrate'],
                'clickrate': predictions['Clickrate'],
                'optoutrate': predictions['Optoutrate'],
                'combined_score': combined_score
            })
    
    # Sort by combined score (highest first)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return results

def display_batch_results(results):
    """Display batch prediction results"""
    if not results:
        st.error("No batch prediction results to display.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create formatted version for display
    display_df = df.copy()
    display_df['openrate'] = display_df['openrate'].map(lambda x: f"{x:.2%}")
    display_df['clickrate'] = display_df['clickrate'].map(lambda x: f"{x:.2%}")
    display_df['optoutrate'] = display_df['optoutrate'].map(lambda x: f"{x:.2%}")
    display_df['combined_score'] = display_df['combined_score'].map(lambda x: f"{x:.2f}")
    
    # Create columns for ranking display
    display_df.insert(0, 'rank', range(1, len(display_df) + 1))
    
    # Display table
    st.dataframe(display_df, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Open Rate", f"{df['openrate'].mean():.2%}")
    
    with col2:
        st.metric("Avg Click Rate", f"{df['clickrate'].mean():.2%}")
    
    with col3:
        st.metric("Avg Opt-out Rate", f"{df['optoutrate'].mean():.2%}")
    
    # Create download button
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, sep=';')
    csv_str = csv_buffer.getvalue()
    
    st.download_button(
        label="Download Results CSV",
        data=csv_str,
        file_name="batch_prediction_results.csv",
        mime="text/csv"
    )
    
    # Visualize results
    st.subheader("Open Rate Distribution")
    
    fig = px.histogram(
        df,
        x='openrate',
        nbins=20,
        title='Distribution of Predicted Open Rates',
        labels={'openrate': 'Open Rate'}
    )
    
    fig.update_xaxes(tickformat='.1%')
    
    st.plotly_chart(fig, use_container_width=True)

def create_kpi_dashboard(df):
    """Create a KPI dashboard with visualizations"""
    st.header("KPI Dashboard")
    
    # Calculate overall metrics
    avg_open = df['Openrate'].mean()
    avg_click = df['Clickrate'].mean()
    avg_optout = df['Optoutrate'].mean()
    total_campaigns = len(df)
    total_sendouts = df['Sendouts'].sum()
    
    # Display metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Open Rate", f"{avg_open:.2%}")
    col2.metric("Avg Click Rate", f"{avg_click:.2%}")
    col3.metric("Avg Opt-out Rate", f"{avg_optout:.2%}")
    col4.metric("Total Campaigns", f"{total_campaigns}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["KPI Trends", "Category Analysis", "Subject Line Analysis"])
    
    with tab1:
        st.subheader("KPI Distribution")
        
        # Create histogram of open rates
        fig = px.histogram(
            df,
            x='Openrate',
            nbins=20,
            title='Distribution of Open Rates',
            labels={'Openrate': 'Open Rate'}
        )
        
        fig.update_xaxes(tickformat='.1%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series if available
        if 'Date' in df.columns and df['Date'].nunique() > 1:
            st.subheader("KPI Trends Over Time")
            
            # Group by month
            df['month'] = df['Date'].dt.to_period('M')
            monthly = df.groupby('month').agg({
                'Openrate': 'mean',
                'Clickrate': 'mean',
                'Optoutrate': 'mean',
                'Sendouts': 'sum'
            }).reset_index()
            
            monthly['month'] = monthly['month'].astype(str)
            
            # Create time series chart
            fig = px.line(
                monthly,
                x='month',
                y=['Openrate', 'Clickrate', 'Optoutrate'],
                title='KPI Trends by Month',
                labels={'value': 'Rate', 'variable': 'Metric', 'month': 'Month'}
            )
            
            fig.update_yaxes(tickformat='.1%')
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Performance by Category")
        
        # Select category to analyze
        category = st.selectbox(
            "Select category",
            options=["Dialog", "Syfte", "Product"]
        )
        
        # Group by selected category
        category_stats = df.groupby(category).agg({
            'Openrate': 'mean',
            'Clickrate': 'mean',
            'Optoutrate': 'mean',
            'Sendouts': 'sum',
            'Subject': 'count'
        }).reset_index()
        
        category_stats = category_stats.rename(columns={'Subject': 'Campaign Count'})
        category_stats = category_stats.sort_values('Openrate', ascending=False)
        
        # Format for display
        display_stats = category_stats.copy()
        for col in ['Openrate', 'Clickrate', 'Optoutrate']:
            display_stats[col] = display_stats[col].map(lambda x: f"{x:.2%}")
        
        # Display table
        st.dataframe(display_stats, use_container_width=True)
        
        # Create bar chart
        fig = px.bar(
            category_stats,
            y=category,
            x='Openrate',
            title=f'Open Rate by {category}',
            labels={'Openrate': 'Open Rate'},
            orientation='h',
            text_auto='.1%'
        )
        
        fig.update_xaxes(tickformat='.1%')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Subject Line Analysis")
        
        # Subject line length vs open rate
        fig = px.scatter(
            df,
            x='Subject_length',
            y='Openrate',
            title='Subject Length vs Open Rate',
            labels={'Subject_length': 'Subject Length (characters)', 'Openrate': 'Open Rate'},
            trendline='ols'
        )
        
        fig.update_yaxes(tickformat='.1%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Binary feature impact
        binary_features = [
            ('Subject_has_question', 'Has Question Mark'),
            ('Subject_has_exclamation', 'Has Exclamation Mark'),
            ('Subject_has_number', 'Has Number')
        ]
        
        for feature, display_name in binary_features:
            if feature in df.columns:
                # Group by binary feature
                feature_stats = df.groupby(feature).agg({
                    'Openrate': 'mean',
                    'Clickrate': 'mean',
                    'Optoutrate': 'mean',
                    'Subject': 'count'
                }).reset_index()
                
                feature_stats[feature] = feature_stats[feature].map({0: 'No', 1: 'Yes'})
                
                # Create grouped bar chart
                fig = px.bar(
                    feature_stats,
                    x=feature,
                    y=['Openrate', 'Clickrate', 'Optoutrate'],
                    title=f'KPIs by Subject Line {display_name}',
                    labels={'value': 'Rate', 'variable': 'Metric'},
                    barmode='group',
                    text_auto='.1%'
                )
                
                fig.update_yaxes(tickformat='.1%')
                
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Set page config
    st.set_page_config(
        page_title="Email Campaign Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
        .main .block-container {max-width: 1200px; padding-top: 2rem;}
        h1 {margin-bottom: 1rem;}
        .stMetric {border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: 1rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 1rem;}
        .stTabs [data-baseweb="tab"] {padding-top: 0.5rem; padding-bottom: 0.5rem;}
    </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.title("Multi-KPI Email Campaign Optimizer")
    
    # Sidebar with model settings
    st.sidebar.header("Model Settings")
    
    force_retrain = st.sidebar.checkbox("Force retrain models", value=False)
    if st.sidebar.button("Retrain Models"):
        force_retrain = True
    
    # Load data
    delivery_df, customer_df = load_data()
    
    # Check if data is loaded
    if delivery_df is None or customer_df is None:
        st.info("Please upload data files first.")
        return
    
    try:
        # Preprocess data
        df = preprocess_data(delivery_df, customer_df)
        
        # Train models
        models = train_models(df, force_retrain)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Campaign Optimizer", 
            "Batch Processing", 
            "KPI Dashboard", 
            "Model Insights"
        ])
        
        # Tab 1: Campaign Optimizer
        with tab1:
            st.header("Optimize Your Email Campaign")
            
            # Create a form for input
            with st.form(key="campaign_form"):
                # Campaign metadata
                st.subheader("Campaign Settings")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Dialog
                    dialog_options = [(key, val[1]) for key, val in DIALOG_VALUES.items()]
                    dialog_display = st.selectbox(
                        "Dialog",
                        options=[label for _, label in dialog_options]
                    )
                    dialog = next(code for code, label in dialog_options if label == dialog_display)
                
                with col2:
                    # Syfte
                    syfte_options = [(key, val[1]) for key, val in SYFTE_VALUES.items()]
                    syfte_display = st.selectbox(
                        "Syfte",
                        options=[label for _, label in syfte_options]
                    )
                    syfte = next(code for code, label in syfte_options if label == syfte_display)
                
                with col3:
                    # Product
                    product_options = [(key, val[1]) for key, val in PRODUKT_VALUES.items()]
                    product_display = st.selectbox(
                        "Product",
                        options=[label for _, label in product_options]
                    )
                    product = next(code for code, label in product_options if label == product_display)
                
                # Audience
                st.subheader("Target Audience")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_age = st.number_input("Minimum Age", min_value=18, max_value=100, value=18)
                
                with col2:
                    max_age = st.number_input("Maximum Age", min_value=18, max_value=100, value=100)
                
                # Bolag multiselect (which to exclude)
                excluded_bolag = st.multiselect(
                    "Exclude Bolag",
                    options=list(BOLAG_VALUES.keys())
                )
                included_bolag = [b for b in BOLAG_VALUES.keys() if b not in excluded_bolag]
                
                # Email content
                st.subheader("Email Content")
                subject = st.text_input("Subject Line")
                preheader = st.text_input("Preheader")
                
                # Options
                col1, col2 = st.columns(2)
                with col1:
                    generate_ai = st.checkbox("Generate alternatives with AI", value=True)
                
                # Submit button
                submit_button = st.form_submit_button("Predict KPIs")
            
            # Process form submission
            if submit_button and subject:
                # Predict KPIs
                predictions = predict_kpis(
                    models,
                    subject,
                    preheader,
                    dialog,
                    syfte,
                    product,
                    min_age,
                    max_age,
                    included_bolag
                )
                
                if predictions:
                    # Display predictions
                    st.subheader("Predicted KPIs")
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Open Rate", f"{predictions['Openrate']:.2%}")
                    col2.metric("Click Rate", f"{predictions['Clickrate']:.2%}")
                    col3.metric("Opt-out Rate", f"{predictions['Optoutrate']:.2%}")
                    
                    # Generate AI alternatives if requested
                    if generate_ai:
                        with st.spinner("Generating alternatives with AI..."):
                            suggestions = get_subject_suggestions(
                                subject,
                                preheader,
                                dialog,
                                syfte,
                                product,
                                predictions,
                                min_age,
                                max_age,
                                included_bolag
                            )
                            
                            if suggestions:
                                # Compare alternatives
                                comparison_results = compare_suggestions(
                                    models,
                                    subject,
                                    preheader,
                                    suggestions,
                                    dialog,
                                    syfte,
                                    product,
                                    min_age,
                                    max_age,
                                    included_bolag
                                )
                                
                                # Display comparison
                                st.subheader("A/B/C/D Testing Results")
                                display_comparative_results(comparison_results)
                            else:
                                st.error("Failed to generate alternatives. Please check your Groq API key.")
        
                
        # Tab 2: KPI Dashboard
        with tab2:
            create_kpi_dashboard(df)
            
        # Tab 3: Batch Processing
        with tab3:
            st.header("Batch KPI Prediction")
            
            # Create tabs for different input methods
            input_tab1, input_tab2 = st.tabs(["Upload CSV", "Enter Manually"])
            
            with input_tab1:
                uploaded_file = st.file_uploader("Upload subject lines CSV", type=["csv"], key="batch_csv")
                
                if uploaded_file:
                    # Read CSV
                    try:
                        batch_df = pd.read_csv(uploaded_file, sep=None, engine='python')
                        
                        # Check for required column
                        if 'Subject' not in batch_df.columns:
                            st.error("CSV must contain a 'Subject' column.")
                        else:
                            # Show preview
                            st.subheader("Preview")
                            st.dataframe(batch_df.head(5), use_container_width=True)
                            
                            # Set parameters
                            st.subheader("Campaign Settings")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Dialog
                                dialog_options = [(key, val[1]) for key, val in DIALOG_VALUES.items()]
                                dialog_display = st.selectbox(
                                    "Dialog",
                                    options=[label for _, label in dialog_options],
                                    key="batch_dialog"
                                )
                                batch_dialog = next(code for code, label in dialog_options if label == dialog_display)
                            
                            with col2:
                                # Syfte
                                syfte_options = [(key, val[1]) for key, val in SYFTE_VALUES.items()]
                                syfte_display = st.selectbox(
                                    "Syfte",
                                    options=[label for _, label in syfte_options],
                                    key="batch_syfte"
                                )
                                batch_syfte = next(code for code, label in syfte_options if label == syfte_display)
                            
                            with col3:
                                # Product
                                product_options = [(key, val[1]) for key, val in PRODUKT_VALUES.items()]
                                product_display = st.selectbox(
                                    "Product",
                                    options=[label for _, label in product_options],
                                    key="batch_product"
                                )
                                batch_product = next(code for code, label in product_options if label == product_display)
                            
                            # Audience
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                batch_min_age = st.number_input("Minimum Age", min_value=18, max_value=100, value=18, key="batch_min_age")
                            
                            with col2:
                                batch_max_age = st.number_input("Maximum Age", min_value=18, max_value=100, value=100, key="batch_max_age")
                            
                            # Bolag multiselect
                            batch_excluded_bolag = st.multiselect(
                                "Exclude Bolag",
                                options=list(BOLAG_VALUES.keys()),
                                key="batch_excluded_bolag"
                            )
                            batch_included_bolag = [b for b in BOLAG_VALUES.keys() if b not in batch_excluded_bolag]
                            
                            # Handle preheader column
                            has_preheader = 'Preheader' in batch_df.columns
                            
                            # Run batch prediction
                            if st.button("Run Batch Prediction"):
                                with st.spinner("Processing batch prediction..."):
                                    # Get subject and preheader lists
                                    subjects = batch_df['Subject'].tolist()
                                    preheaders = batch_df['Preheader'].tolist() if has_preheader else [''] * len(subjects)
                                    
                                    # Run prediction
                                    batch_results = batch_predict(
                                        models,
                                        subjects,
                                        preheaders,
                                        batch_dialog,
                                        batch_syfte,
                                        batch_product,
                                        batch_min_age,
                                        batch_max_age,
                                        batch_included_bolag
                                    )
                                    
                                    # Display results
                                    st.subheader("Batch Prediction Results")
                                    display_batch_results(batch_results)
                    
                    except Exception as e:
                        st.error(f"Error processing CSV file: {e}")
            
            with input_tab2:
                # Text area for manual entry
                st.subheader("Enter Subject Lines")
                manual_subjects = st.text_area(
                    "Enter subject lines (one per line)",
                    height=150
                )
                
                manual_preheaders = st.text_area(
                    "Enter preheaders (one per line, must match number of subject lines)",
                    height=150
                )
                
                # Campaign settings
                st.subheader("Campaign Settings")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Dialog
                    dialog_options = [(key, val[1]) for key, val in DIALOG_VALUES.items()]
                    dialog_display = st.selectbox(
                        "Dialog",
                        options=[label for _, label in dialog_options],
                        key="manual_dialog"
                    )
                    manual_dialog = next(code for code, label in dialog_options if label == dialog_display)
                
                with col2:
                    # Syfte
                    syfte_options = [(key, val[1]) for key, val in SYFTE_VALUES.items()]
                    syfte_display = st.selectbox(
                        "Syfte",
                        options=[label for _, label in syfte_options],
                        key="manual_syfte"
                    )
                    manual_syfte = next(code for code, label in syfte_options if label == syfte_display)
                
                with col3:
                    # Product
                    product_options = [(key, val[1]) for key, val in PRODUKT_VALUES.items()]
                    product_display = st.selectbox(
                        "Product",
                        options=[label for _, label in product_options],
                        key="manual_product"
                    )
                    manual_product = next(code for code, label in product_options if label == product_display)
                
                # Audience
                col1, col2 = st.columns(2)
                
                with col1:
                    manual_min_age = st.number_input("Minimum Age", min_value=18, max_value=100, value=18, key="manual_min_age")
                
                with col2:
                    manual_max_age = st.number_input("Maximum Age", min_value=18, max_value=100, value=100, key="manual_max_age")
                
                # Bolag multiselect
                manual_excluded_bolag = st.multiselect(
                    "Exclude Bolag",
                    options=list(BOLAG_VALUES.keys()),
                    key="manual_excluded_bolag"
                )
                manual_included_bolag = [b for b in BOLAG_VALUES.keys() if b not in manual_excluded_bolag]
                
                # Run prediction
                if st.button("Run Manual Batch Prediction"):
                    if not manual_subjects.strip():
                        st.error("Please enter at least one subject line.")
                    else:
                        with st.spinner("Processing batch prediction..."):
                            # Parse input
                            subject_lines = [line.strip() for line in manual_subjects.strip().split('\n') if line.strip()]
                            preheader_lines = [line.strip() for line in manual_preheaders.strip().split('\n') if line.strip()]
                            
                            # Check lengths match
                            if preheader_lines and len(preheader_lines) != len(subject_lines):
                                st.error(f"Number of preheaders ({len(preheader_lines)}) must match number of subject lines ({len(subject_lines)}).")
                            else:
                                # Pad preheaders if needed
                                if not preheader_lines:
                                    preheader_lines = [''] * len(subject_lines)
                                
                                # Run prediction
                                batch_results = batch_predict(
                                    models,
                                    subject_lines,
                                    preheader_lines,
                                    manual_dialog,
                                    manual_syfte,
                                    manual_product,
                                    manual_min_age,
                                    manual_max_age,
                                    manual_included_bolag
                                )
                                
                                # Display results
                                st.subheader("Batch Prediction Results")
                                display_batch_results(batch_results)
        
        # Tab 4: Model Insights
        with tab4:
            st.header("Model Performance & Insights")
            
            # Show model metrics
            st.subheader("Model Performance")
            metrics_data = []
            
            for kpi in ['Openrate', 'Clickrate', 'Optoutrate']:
                if kpi in models and 'metrics' in models[kpi]:
                    metrics = models[kpi]['metrics']
                    metrics_data.append({
                        'KPI': kpi,
                        'RMSE': metrics.get('rmse', 0),
                        'R²': metrics.get('r2', 0)
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Create columns for metrics
                cols = st.columns(len(metrics_data))
                
                for i, (col, row) in enumerate(zip(cols, metrics_data)):
                    col.metric(
                        f"{row['KPI']} Model",
                        f"R² = {row['R²']:.4f}",
                        f"RMSE = {row['RMSE']:.4f}"
                    )
                
                # Create bar chart of R² scores
                fig = px.bar(
                    metrics_df,
                    x='KPI',
                    y='R²',
                    title='Model R² Scores',
                    labels={'R²': 'R² Score'},
                    text_auto='.4f'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Select KPI model to analyze
            kpi_select = st.selectbox(
                "Select KPI model",
                options=['Openrate', 'Clickrate', 'Optoutrate']
            )
            
            if kpi_select in models and 'model' in models[kpi_select]:
                model = models[kpi_select]['model']
                
                # Get feature importance
                feature_importance = pd.DataFrame({
                    'Feature': models['feature_names'],
                    'Importance': model.feature_importances_
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Display top N features
                top_n = st.slider("Number of top features to show", 5, 30, 15)
                top_features = feature_importance.head(top_n)
                
                # Create horizontal bar chart
                fig = px.bar(
                    top_features,
                    y='Feature',
                    x='Importance',
                    title=f'Top {top_n} Features for {kpi_select}',
                    labels={'Importance': 'Feature Importance'},
                    orientation='h'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation with KPI for numeric features
                st.subheader("Feature Correlations")
                
                # Get numeric features
                numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
                valid_features = [f for f in numeric_features if f not in ['Openrate', 'Clickrate', 'Optoutrate', 'Sendouts', 'Opens', 'Clicks', 'Optouts']]
                
                if valid_features:
                    # Calculate correlations
                    correlations = []
                    
                    for feature in valid_features:
                        corr = df[feature].corr(df[kpi_select])
                        correlations.append({
                            'Feature': feature,
                            'Correlation': corr
                        })
                    
                    corr_df = pd.DataFrame(correlations)
                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                    
                    # Display top correlations
                    fig = px.bar(
                        corr_df.head(10),
                        y='Feature',
                        x='Correlation',
                        title=f'Top Correlations with {kpi_select}',
                        labels={'Correlation': 'Correlation Coefficient'},
                        orientation='h'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show scatter plot for top correlated feature
                    if not corr_df.empty:
                        top_feature = corr_df.iloc[0]['Feature']
                        
                        fig = px.scatter(
                            df,
                            x=top_feature,
                            y=kpi_select,
                            title=f'{top_feature} vs {kpi_select}',
                            trendline='ols'
                        )
                        
                        fig.update_yaxes(tickformat='.1%')
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()