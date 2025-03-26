import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from dotenv import load_dotenv
import os
import joblib
import groq
import datetime
import logging
import traceback
import shutil
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import textstat
import re

# Download necessary NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass  # Handle offline case gracefully

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KPI_Predictor')

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Constants and Configuration ---
MODEL_BASE_DIR = 'models'
DOCS_DIR = 'Docs'

# Generate current model version based on date (YY.MM.DD format)
def get_current_model_version():
    """
    Generate a model version based on the current date in YY.MM.DD format
    """
    today = datetime.datetime.now()
    return f"{today.strftime('%y.%m.%d')}"

CURRENT_MODEL_VERSION = get_current_model_version()  # e.g., 25.03.25 for March 25, 2025

# --- Enum Constants ---
# [Same as original code, keeping these constants unchanged]
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

# KPI Types for multi-model support
KPI_TYPES = ['openrate', 'clickrate', 'optoutrate']

# Ensure directories exist
for directory in [MODEL_BASE_DIR, DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- New Text Analysis Functions ---
def preprocess_text(text):
    """
    Preprocess text for NLP analysis
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_sentiment_features(text):
    """
    Extract sentiment features from text using VADER sentiment analyzer
    """
    if not isinstance(text, str) or not text.strip():
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    
    try:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}

def calculate_readability_metrics(text):
    """
    Calculate readability metrics for the given text
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'difficult_words': 0
        }
    
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'difficult_words': textstat.difficult_words(text)
        }
    except Exception as e:
        logger.error(f"Error calculating readability metrics: {str(e)}")
        return {
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'difficult_words': 0
        }

def extract_text_patterns(text):
    """
    Extract patterns from text like numbers, currency symbols, etc.
    """
    if not isinstance(text, str):
        return {
            'has_numbers': 0,
            'has_currency': 0,
            'has_percentage': 0,
            'has_url': 0,
            'has_emoji': 0
        }
    
    patterns = {
        'has_numbers': 1 if re.search(r'\d', text) else 0,
        'has_currency': 1 if re.search(r'[$€£kr]', text) else 0,
        'has_percentage': 1 if re.search(r'%', text) else 0,
        'has_url': 1 if re.search(r'https?://\S+|www\.\S+', text) else 0,
        'has_emoji': 1 if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text) else 0
    }
    
    return patterns

def extract_time_features(date_str):
    """
    Extract time-based features from date string
    """
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y/%m/%d %H:%M')
        return {
            'hour': date_obj.hour,
            'day_of_week': date_obj.weekday(),
            'month': date_obj.month,
            'is_weekend': 1 if date_obj.weekday() >= 5 else 0,
            'is_morning': 1 if 5 <= date_obj.hour < 12 else 0,
            'is_afternoon': 1 if 12 <= date_obj.hour < 17 else 0,
            'is_evening': 1 if 17 <= date_obj.hour < 21 else 0,
            'is_night': 1 if date_obj.hour >= 21 or date_obj.hour < 5 else 0
        }
    except Exception as e:
        logger.error(f"Error extracting time features: {str(e)}")
        return {
            'hour': 0,
            'day_of_week': 0,
            'month': 0,
            'is_weekend': 0,
            'is_morning': 0,
            'is_afternoon': 0,
            'is_evening': 0,
            'is_night': 0
        }

# --- Age Group Utils ---
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

# --- Remaining utility functions from original code ---
def process_data_for_age_heatmap(delivery_data, customer_data):
    """
    Process data for age group heatmap
    """
    # [Function body remains the same as in original code]
    # This function is detailed and doesn't need changes for the multi-model approach
    pass

def create_age_heatmap(heatmap_data, metric, title, cmap='viridis', figsize=(12, 6)):
    """
    Create a heatmap for a specific metric by age group
    """
    # [Function body remains the same as in original code]
    pass

def create_interactive_heatmap(data, metric, title, is_percentage=True, colorscale='Viridis'):
    """
    Create an interactive heatmap using Plotly for a specific metric
    """
    # [Function body remains the same as in original code]
    pass

# --- Model Version Management ---
def get_model_filename(kpi_type='openrate', version=CURRENT_MODEL_VERSION):
    """
    Generate model filename based on KPI type and version
    """
    return os.path.join(MODEL_BASE_DIR, f"{kpi_type}_model_v{version}.pkl")

def get_model_metadata_filename(kpi_type='openrate', version=CURRENT_MODEL_VERSION):
    """
    Generate metadata filename based on KPI type and version
    """
    return os.path.join(MODEL_BASE_DIR, f"{kpi_type}_model_v{version}_metadata.yaml")

def save_model_metadata(metadata, kpi_type='openrate', version=CURRENT_MODEL_VERSION):
    """
    Save model metadata to a YAML file
    """
    try:
        metadata_file = get_model_metadata_filename(kpi_type, version)
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"{kpi_type.capitalize()} model metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving {kpi_type} model metadata: {str(e)}")
        logger.error(traceback.format_exc())

def load_model_metadata(kpi_type='openrate', version=CURRENT_MODEL_VERSION):
    """
    Load model metadata from a YAML file
    """
    try:
        metadata_file = get_model_metadata_filename(kpi_type, version)
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
            return metadata
        else:
            logger.warning(f"{kpi_type.capitalize()} model metadata file not found: {metadata_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading {kpi_type} model metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def list_available_models(kpi_type='openrate'):
    """
    List all available model versions for a specific KPI type
    """
    try:
        model_files = [f for f in os.listdir(MODEL_BASE_DIR) if f.startswith(f'{kpi_type}_model_v') and f.endswith('.pkl')]
        versions = [f.split('_v')[1].replace('.pkl', '') for f in model_files]
        versions.sort(key=lambda s: [int(u) for u in s.split('.')])
        return versions
    except Exception as e:
        logger.error(f"Error listing available {kpi_type} models: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# --- Data Loading and Error Handling ---
def load_data(delivery_file='../Data/delivery_data.csv', customer_file='../Data/customer_data.csv'):
    """
    Load delivery and customer data with error handling
    """
    try:
        logger.info(f"Loading data from {delivery_file} and {customer_file}")
        delivery_data = pd.read_csv(delivery_file, sep=';', encoding='utf-8')
        customer_data = pd.read_csv(customer_file, sep=';', encoding='utf-8')
        
        # Data validation
        required_delivery_cols = ['Subject', 'Dialog', 'Syfte', 'Product', 'Opens', 'Sendouts', 'Clicks', 'Optouts', 'InternalName', 'Date']
        required_customer_cols = ['InternalName', 'Age', 'Bolag']
        
        # Add Preheader if it exists, otherwise create empty column
        if 'Preheader' not in delivery_data.columns:
            delivery_data['Preheader'] = ""
            logger.warning("Preheader column not found in delivery data, creating empty column")
        else:
            required_delivery_cols.append('Preheader')
            
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

# --- Enhanced Feature Engineering ---
def engineer_features(delivery_data, customer_data, include_preheader=True, include_nlp=True, include_time=True):
    """
    Perform enhanced feature engineering on the dataset
    
    Parameters:
    delivery_data (pd.DataFrame): Delivery data
    customer_data (pd.DataFrame): Customer data
    include_preheader (bool): Whether to include preheader features
    include_nlp (bool): Whether to include advanced NLP features
    include_time (bool): Whether to include time-based features
    
    Returns:
    tuple: features DataFrame dict, targets dict, and a dictionary of feature metadata
    """
    try:
        logger.info("Starting enhanced feature engineering")
        
        # Calculate KPIs
        delivery_data['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        delivery_data['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
        delivery_data['Optoutrate'] = delivery_data['Optouts'] / delivery_data['Sendouts']
        
        # Check for NaN values and fix them
        for kpi in ['Openrate', 'Clickrate', 'Optoutrate']:
            delivery_data[kpi] = delivery_data[kpi].fillna(0)
        
        # Basic features (similar to original code)
        # Aggregate age stats
        age_stats = customer_data.groupby('InternalName')['Age'].agg(['min', 'max']).reset_index()
        age_stats.columns = ['InternalName', 'Min_age', 'Max_age']
        delivery_data = delivery_data.merge(age_stats, on='InternalName', how='left')
        
        # Aggregate Bolag as binary features
        bolag_dummies = pd.get_dummies(customer_data['Bolag'], prefix='Bolag')
        customer_data_with_dummies = pd.concat([customer_data, bolag_dummies], axis=1)
        bolag_features = customer_data_with_dummies.groupby('InternalName')[bolag_dummies.columns].max().reset_index()
        delivery_data = delivery_data.merge(bolag_features, on='InternalName', how='left')
        
        # Legacy text features
        delivery_data['Subject_length'] = delivery_data['Subject'].str.len()
        delivery_data['Subject_num_words'] = delivery_data['Subject'].str.split().str.len()
        delivery_data['Subject_has_exclamation'] = delivery_data['Subject'].str.contains('!').astype(int)
        delivery_data['Subject_has_question'] = delivery_data['Subject'].str.contains(r'\?', regex=True).astype(int)
        
        # Enhanced text features for subject
        delivery_data['Subject_caps_ratio'] = delivery_data['Subject'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        delivery_data['Subject_avg_word_len'] = delivery_data['Subject'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Count specific characters that might impact open rates
        delivery_data['Subject_num_special_chars'] = delivery_data['Subject'].apply(
            lambda x: sum(1 for c in str(x) if c in '!?%$€£#@*&')
        )
        
        # Extract first and last words of subject (useful for headlines)
        delivery_data['Subject_first_word_len'] = delivery_data['Subject'].apply(
            lambda x: len(str(x).split()[0]) if len(str(x).split()) > 0 else 0
        )
        
        delivery_data['Subject_last_word_len'] = delivery_data['Subject'].apply(
            lambda x: len(str(x).split()[-1]) if len(str(x).split()) > 0 else 0
        )
        
        # Add preheader features if requested
        if include_preheader:
            # Basic preheader features
            delivery_data['Preheader_length'] = delivery_data['Preheader'].str.len()
            delivery_data['Preheader_num_words'] = delivery_data['Preheader'].str.split().str.len()
            delivery_data['Preheader_has_exclamation'] = delivery_data['Preheader'].str.contains('!').astype(int)
            delivery_data['Preheader_has_question'] = delivery_data['Preheader'].str.contains(r'\?', regex=True).astype(int)
            
            # Enhanced preheader features
            delivery_data['Preheader_caps_ratio'] = delivery_data['Preheader'].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
            )
            
            delivery_data['Preheader_avg_word_len'] = delivery_data['Preheader'].apply(
                lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
            )
            
            delivery_data['Preheader_num_special_chars'] = delivery_data['Preheader'].apply(
                lambda x: sum(1 for c in str(x) if c in '!?%$€£#@*&')
            )
            
            # Relationship between subject and preheader
            delivery_data['Subject_preheader_length_ratio'] = delivery_data['Subject_length'] / delivery_data['Preheader_length'].replace(0, 1)
            delivery_data['Subject_preheader_words_ratio'] = delivery_data['Subject_num_words'] / delivery_data['Preheader_num_words'].replace(0, 1)
        
        # Add advanced NLP features if requested
        if include_nlp:
            # Sentiment analysis for subject
            subject_sentiment = delivery_data['Subject'].apply(extract_sentiment_features)
            delivery_data['Subject_sentiment_compound'] = subject_sentiment.apply(lambda x: x['compound'])
            delivery_data['Subject_sentiment_positive'] = subject_sentiment.apply(lambda x: x['pos'])
            delivery_data['Subject_sentiment_negative'] = subject_sentiment.apply(lambda x: x['neg'])
            
            # Readability metrics for subject
            subject_readability = delivery_data['Subject'].apply(calculate_readability_metrics)
            delivery_data['Subject_flesch_reading_ease'] = subject_readability.apply(lambda x: x['flesch_reading_ease'])
            delivery_data['Subject_flesch_kincaid_grade'] = subject_readability.apply(lambda x: x['flesch_kincaid_grade'])
            delivery_data['Subject_difficult_words'] = subject_readability.apply(lambda x: x['difficult_words'])
            
            # Text patterns for subject
            subject_patterns = delivery_data['Subject'].apply(extract_text_patterns)
            delivery_data['Subject_has_numbers'] = subject_patterns.apply(lambda x: x['has_numbers'])
            delivery_data['Subject_has_currency'] = subject_patterns.apply(lambda x: x['has_currency'])
            delivery_data['Subject_has_percentage'] = subject_patterns.apply(lambda x: x['has_percentage'])
            
            if include_preheader:
                # Sentiment analysis for preheader
                preheader_sentiment = delivery_data['Preheader'].apply(extract_sentiment_features)
                delivery_data['Preheader_sentiment_compound'] = preheader_sentiment.apply(lambda x: x['compound'])
                delivery_data['Preheader_sentiment_positive'] = preheader_sentiment.apply(lambda x: x['pos'])
                delivery_data['Preheader_sentiment_negative'] = preheader_sentiment.apply(lambda x: x['neg'])
                
                # Readability metrics for preheader
                preheader_readability = delivery_data['Preheader'].apply(calculate_readability_metrics)
                delivery_data['Preheader_flesch_reading_ease'] = preheader_readability.apply(lambda x: x['flesch_reading_ease'])
                delivery_data['Preheader_flesch_kincaid_grade'] = preheader_readability.apply(lambda x: x['flesch_kincaid_grade'])
                
                # Sentiment agreement between subject and preheader
                delivery_data['Subject_preheader_sentiment_agreement'] = (
                    (delivery_data['Subject_sentiment_compound'] > 0) & 
                    (delivery_data['Preheader_sentiment_compound'] > 0)
                ).astype(int) + (
                    (delivery_data['Subject_sentiment_compound'] < 0) & 
                    (delivery_data['Preheader_sentiment_compound'] < 0)
                ).astype(int)
        
        # Add time-based features if requested
        if include_time and 'Date' in delivery_data.columns:
            time_features = delivery_data['Date'].apply(extract_time_features)
            delivery_data['Hour'] = time_features.apply(lambda x: x['hour'])
            delivery_data['DayOfWeek'] = time_features.apply(lambda x: x['day_of_week'])
            delivery_data['Month'] = time_features.apply(lambda x: x['month'])
            delivery_data['IsWeekend'] = time_features.apply(lambda x: x['is_weekend'])
            delivery_data['IsMorning'] = time_features.apply(lambda x: x['is_morning'])
            delivery_data['IsAfternoon'] = time_features.apply(lambda x: x['is_afternoon'])
            delivery_data['IsEvening'] = time_features.apply(lambda x: x['is_evening'])
        
        # Define feature columns
        categorical_features = ['Dialog', 'Syfte', 'Product']
        
        # Define numerical features based on feature sets
        basic_numerical_features = [
            'Min_age', 'Max_age', 
            'Subject_length', 'Subject_num_words', 'Subject_has_exclamation', 'Subject_has_question',
            'Subject_caps_ratio', 'Subject_avg_word_len', 'Subject_num_special_chars',
            'Subject_first_word_len', 'Subject_last_word_len'
        ]
        
        preheader_numerical_features = [
            'Preheader_length', 'Preheader_num_words', 
            'Preheader_has_exclamation', 'Preheader_has_question',
            'Preheader_caps_ratio', 'Preheader_avg_word_len', 'Preheader_num_special_chars',
            'Subject_preheader_length_ratio', 'Subject_preheader_words_ratio'
        ]
        
        nlp_numerical_features = [
            'Subject_sentiment_compound', 'Subject_sentiment_positive', 'Subject_sentiment_negative',
            'Subject_flesch_reading_ease', 'Subject_flesch_kincaid_grade', 'Subject_difficult_words',
            'Subject_has_numbers', 'Subject_has_currency', 'Subject_has_percentage'
        ]
        
        nlp_preheader_numerical_features = [
            'Preheader_sentiment_compound', 'Preheader_sentiment_positive', 'Preheader_sentiment_negative',
            'Preheader_flesch_reading_ease', 'Preheader_flesch_kincaid_grade',
            'Subject_preheader_sentiment_agreement'
        ]
        
        time_numerical_features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 
            'IsMorning', 'IsAfternoon', 'IsEvening'
        ]
        
        # Combine feature sets based on flags
        numerical_features = basic_numerical_features.copy()
        
        if include_preheader:
            numerical_features.extend(preheader_numerical_features)
            
        if include_nlp:
            numerical_features.extend(nlp_numerical_features)
            if include_preheader:
                numerical_features.extend(nlp_preheader_numerical_features)
                
        if include_time and 'Date' in delivery_data.columns:
            numerical_features.extend(time_numerical_features)
        
        # Filter to keep only columns that exist in the dataset
        numerical_features = [f for f in numerical_features if f in delivery_data.columns]
        
        bolag_features_list = [col for col in delivery_data.columns if col.startswith('Bolag_')]
        
        # Generate categorical dummies
        dummy_df = pd.get_dummies(delivery_data[categorical_features])
        
        # Create mappings for UI
        dummy_dialog_map = {dialog: f'Dialog_{dialog}' for dialog in delivery_data['Dialog'].unique()}
        dummy_syfte_map = {syfte: f'Syfte_{syfte}' for syfte in delivery_data['Syfte'].unique()}
        dummy_product_map = {product: f'Product_{product}' for product in delivery_data['Product'].unique()}
        
        # Prepare features
        features = pd.concat([
            dummy_df,
            delivery_data[numerical_features],
            delivery_data[bolag_features_list].fillna(0).astype(int)
        ], axis=1)
        
        # Target variables for each KPI
        targets = {
            'openrate': delivery_data['Openrate'],
            'clickrate': delivery_data['Clickrate'],
            'optoutrate': delivery_data['Optoutrate']
        }
        
        # Metadata for documentation
        feature_metadata = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'bolag_features': bolag_features_list,
            'dummy_dialog_map': dummy_dialog_map,
            'dummy_syfte_map': dummy_syfte_map,
            'dummy_product_map': dummy_product_map,
            'include_preheader': include_preheader,
            'include_nlp': include_nlp,
            'include_time': include_time
        }
        
        logger.info(f"Enhanced feature engineering completed - Features: {features.shape}")
        
        return {'all': features}, targets, feature_metadata
        
    except Exception as e:
        logger.error(f"Error in enhanced feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error in enhanced feature engineering: {str(e)}")
        raise

# --- Model Training and Validation ---
def train_model(X_train, y_train, kpi_type='openrate', sample_weights=None, params=None):
    """
    Train an XGBoost model for a specific KPI with the given parameters
    """
    try:
        if params is None:
            # Default parameters for different KPI types
            default_params = {
                'openrate': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'reg_lambda': 1.0,
                    'random_state': 42
                },
                'clickrate': {
                    'n_estimators': 120,
                    'learning_rate': 0.08,
                    'max_depth': 4,
                    'reg_lambda': 1.2,
                    'random_state': 42
                },
                'optoutrate': {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'reg_lambda': 1.5,
                    'random_state': 42
                }
            }
            
            params = default_params.get(kpi_type, default_params['openrate'])
        
        logger.info(f"Training {kpi_type} XGBoost model with parameters: {params}")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return model
    except Exception as e:
        logger.error(f"Error training {kpi_type} model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model_with_params(X_train, y_train, kpi_type='openrate', params=None, sample_weight_config=None):
    """
    Train an XGBoost model for a specific KPI with the given parameters and sample weight configuration
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
    kpi_type : str
        Type of KPI to predict ('openrate', 'clickrate', 'optoutrate')
    params : dict
        Model parameters for XGBRegressor
    sample_weight_config : dict
        Configuration for sample weights including threshold and weight ratio
        
    Returns:
    --------
    XGBRegressor
        Trained model
    """
    try:
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
        logger.info(f"Training {kpi_type} XGBoost model with parameters: {params}")
        
        # Configure sample weights if provided
        sample_weights = None
        if sample_weight_config is not None:
            threshold = sample_weight_config.get('threshold', 0.5)
            weight_high = sample_weight_config.get('weight_high', 2.0)
            weight_low = sample_weight_config.get('weight_low', 1.0)
            
            logger.info(f"Using sample weights for {kpi_type} - threshold: {threshold}, high: {weight_high}, low: {weight_low}")
            sample_weights = np.where(y_train > threshold, weight_high, weight_low)
        
        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return model
    except Exception as e:
        logger.error(f"Error training {kpi_type} model with parameters: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
def evaluate_model(model, X_test, y_test, kpi_type='openrate'):
    """
    Evaluate model performance on test data
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
        
        logger.info(f"{kpi_type.capitalize()} model evaluation - MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")
        
        return metrics, y_pred
    except Exception as e:
        logger.error(f"Error evaluating {kpi_type} model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cross_validate_model(X_train, y_train, kpi_type='openrate', params=None, n_splits=5, sample_weights=None):
    """
    Perform cross-validation on the training data
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
        
        logger.info(f"Cross-validation for {kpi_type} model - Mean R²: {cv_results['r2']['mean']:.4f}, Mean RMSE: {cv_results['rmse']['mean']:.6f}")
        
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation for {kpi_type} model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def tune_hyperparameters(X_train, y_train, kpi_type='openrate', param_grid=None, cv=5, sample_weights=None):
    """
    Tune model hyperparameters using GridSearchCV
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
    kpi_type : str
        Type of KPI to predict ('openrate', 'clickrate', 'optoutrate')
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
    """
    try:
        if param_grid is None:
            # Different parameter grids for different KPI types
            param_grids = {
                'openrate': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'reg_lambda': [0.5, 1.0, 1.5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'clickrate': {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.05, 0.08],
                    'reg_lambda': [0.8, 1.2, 1.6],
                    'subsample': [0.7, 0.9],
                    'colsample_bytree': [0.7, 0.9]
                },
                'optoutrate': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [2, 3, 4],
                    'learning_rate': [0.01, 0.03, 0.05],
                    'reg_lambda': [1.0, 1.5, 2.0],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8]
                }
            }
            
            param_grid = param_grids.get(kpi_type, param_grids['openrate'])
            
        logger.info(f"Starting hyperparameter tuning for {kpi_type} model with param grid: {param_grid}")
        st.info(f"Tuning hyperparameters for {kpi_type} model... This may take some time.")
        
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
        
        logger.info(f"Best parameters for {kpi_type} model: {best_params}")
        logger.info(f"Best RMSE score for {kpi_type} model: {-best_score:.6f}")
        
        return best_params, best_score
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning for {kpi_type} model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Documentation ---
def generate_model_documentation(model, feature_metadata, train_metrics, cv_results, test_metrics, kpi_type='openrate', version=CURRENT_MODEL_VERSION):
    """
    Generate model documentation for a specific KPI model
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create documentation directory
        model_doc_dir = os.path.join(DOCS_DIR, f"{kpi_type}_model_v{version}")
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # General model information
        model_info = {
            'kpi_type': kpi_type,
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
        
        logger.info(f"{kpi_type.capitalize()} model documentation saved to {doc_file}")
        return doc_file
    except Exception as e:
        logger.error(f"Error generating {kpi_type} model documentation: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# --- API Interaction ---
def send_to_groq_api(subject_line, preheader, kpi_predictions, selected_dialog, selected_syfte, selected_product, min_age, max_age, included_bolag):
    """
    Send data to Groq API for subject line and preheader suggestions with improved prompting
    """
    try:
        if not GROQ_API_KEY:
            return {"error": "Groq API key not found. Please set GROQ_API_KEY in .env file."}
        
        # Format KPI predictions nicely
        kpi_str = ""
        for kpi, value in kpi_predictions.items():
            kpi_str += f"- {kpi.capitalize()}: {value:.2%}\n"
        
        prompt = f"""
        I need to create email subject lines and preheaders for a marketing campaign that maximize engagement. 
        
        Current subject line: "{subject_line}"
        Current preheader: "{preheader}"
        
        Predicted KPIs:
        {kpi_str}
        
        Campaign details:
        - Dialog: {selected_dialog}
        - Syfte (Purpose): {selected_syfte}
        - Product: {selected_product}
        - Age range: {min_age} to {max_age}
        - Target regions: {', '.join(included_bolag)}
        
        I want you to analyze why the current subject line and preheader might be performing this way, and then generate THREE alternative versions that could improve all KPIs, with special focus on improving the open rate.
        
        For each alternative, explain briefly why you think it will perform better.
        
        Please generate your suggestions in Swedish since our audience is Swedish.
        
        Return your response as a JSON object with a 'suggestions' field containing an array of objects, each with 'subject', 'preheader', and 'reasoning' fields, like this:
        {{
            "analysis": "Brief analysis of current subject line and preheader performance",
            "suggestions": [
                {{
                    "subject": "First alternative subject line",
                    "preheader": "First alternative preheader",
                    "reasoning": "Why this alternative might perform better"
                }},
                {{
                    "subject": "Second alternative subject line",
                    "preheader": "Second alternative preheader",
                    "reasoning": "Why this alternative might perform better"
                }},
                {{
                    "subject": "Third alternative subject line",
                    "preheader": "Third alternative preheader",
                    "reasoning": "Why this alternative might perform better"
                }}
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
                    {"role": "system", "content": "You are an expert in email marketing optimization and Swedish language. Your task is to generate compelling email subject lines and preheaders in Swedish that maximize open rates, click rates, and minimize opt-out rates."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            },
            verify=False,
            timeout=30  # Add timeout
        )
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_data = response.json()
        content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        
        try:
            suggestions_data = json.loads(content)
            return suggestions_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")
            return {"error": "Failed to parse API response", "raw_content": content}
    
    except requests.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        return {"error": f"API request error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error sending to Groq API: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Main Application ---
def main():
    st.title('Enhanced Sendout KPI Predictor')
    
    # Sidebar for model version selection
    st.sidebar.header("Model Settings")
    
    # KPI model selection
    selected_kpi_type = st.sidebar.selectbox(
        "Select KPI Type",
        options=KPI_TYPES,
        index=0,  # Default to openrate
        help="Select which KPI model to use for predictions"
    )
    
    available_models = list_available_models(selected_kpi_type)
    
    if available_models:
        selected_version = st.sidebar.selectbox(
            f"Select {selected_kpi_type.capitalize()} Model Version",
            options=available_models,
            index=len(available_models) - 1  # Default to latest version
        )
    else:
        selected_version = CURRENT_MODEL_VERSION
        st.sidebar.info(f"No saved {selected_kpi_type} models found. Will train version {CURRENT_MODEL_VERSION} if needed.")
    
    # Feature set options
    st.sidebar.header("Feature Options")
    include_preheader = st.sidebar.checkbox("Include Preheader Analysis", value=True)
    include_nlp = st.sidebar.checkbox("Include Advanced NLP Features", value=True)
    include_time = st.sidebar.checkbox("Include Time-Based Features", value=True)
    
    # Force retrain option
    force_retrain = st.sidebar.checkbox("Force model retraining")
    
    # Model files
    model_files = {}
    for kpi_type in KPI_TYPES:
        model_files[kpi_type] = get_model_filename(kpi_type, selected_version)
    
    # Load data
    try:
        delivery_data, customer_data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # Engineer features
    try:
        features_dict, targets_dict, feature_metadata = engineer_features(
            delivery_data, customer_data,
            include_preheader=include_preheader,
            include_nlp=include_nlp,
            include_time=include_time
        )
    except Exception as e:
        st.error(f"Failed to engineer features: {str(e)}")
        return
    
    # Select features
    features = features_dict['all']
    
    # Initialize models dictionary
    models = {}
    model_loaded = {}
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(['Sendout Prediction', 'Model Results', 'Feature Analysis'])
    
    # Load or train models
    for kpi_type in KPI_TYPES:
        target = targets_dict[kpi_type]
        model_file = model_files[kpi_type]
        
        # Split data for this KPI
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Define default sample weights
        THRESHOLD = 0.5 if kpi_type == 'openrate' else 0.2 if kpi_type == 'clickrate' else 0.01  # Different thresholds for each KPI
        WEIGHT_HIGH = 2.0
        WEIGHT_LOW = 1.0
        sample_weights_train = np.where(y_train > THRESHOLD, WEIGHT_HIGH, WEIGHT_LOW)
        
        model_loaded[kpi_type] = False
        if os.path.exists(model_file) and not force_retrain:
            try:
                models[kpi_type] = joblib.load(model_file)
                logger.info(f"Loaded {kpi_type} model from {model_file}")
                model_loaded[kpi_type] = True
                
                # Validate features
                valid, message = validate_model_features(models[kpi_type], features)
                if not valid:
                    with tab2:
                        st.warning(f"Feature mismatch detected for {kpi_type} model: {message}")
                        st.info("Adapting features to match model expectations...")
                    
                    # Adapt features to match model
                    features_adapted = adapt_features_to_model(models[kpi_type], features)
                    X_train_adapted, X_test_adapted, _, _ = train_test_split(features_adapted, target, test_size=0.2, random_state=42)
                    
                    # Use adapted features for prediction
                    X_train, X_test = X_train_adapted, X_test_adapted
                
                metadata = load_model_metadata(kpi_type, selected_version)
                if metadata:
                    with tab2:
                        st.sidebar.info(f"{kpi_type.capitalize()} model v{selected_version} loaded with {len(models[kpi_type].feature_names_) if hasattr(models[kpi_type], 'feature_names_') else 'unknown'} features")
                        st.sidebar.info(f"Trained on {metadata.get('training_samples', 'unknown')} samples")
                else:
                    with tab2:
                        st.sidebar.info(f"{kpi_type.capitalize()} model v{selected_version} loaded but no metadata found")
            except Exception as e:
                with tab2:
                    st.error(f"Error loading {kpi_type} model: {str(e)}")
                    st.info(f"Training new {kpi_type} model instead...")
                force_retrain = True
        
        if force_retrain or not os.path.exists(model_file):
            try:
                logger.info(f"Training new {kpi_type} model version {selected_version}")
                with tab2:
                    st.info(f"Training new {kpi_type} model version {selected_version}...")
                
                # Train model with default parameters
                params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42
                }
                models[kpi_type] = train_model(X_train, y_train, kpi_type, sample_weights=sample_weights_train, params=params)
                
                # Save model
                joblib.dump(models[kpi_type], model_file)
                logger.info(f"Saved {kpi_type} model to {model_file}")
                
                # Save metadata
                metadata = {
                    'kpi_type': kpi_type,
                    'version': selected_version,
                    'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'training_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'feature_names': X_train.columns.tolist(),
                    'feature_count': X_train.shape[1],
                    'include_preheader': include_preheader,
                    'include_nlp': include_nlp,
                    'include_time': include_time,
                    'model_parameters': params,
                    'sample_weights': {
                        'threshold': THRESHOLD,
                        'weight_high': WEIGHT_HIGH,
                        'weight_low': WEIGHT_LOW
                    }
                }
                save_model_metadata(metadata, kpi_type, selected_version)
                
                with tab2:
                    st.success(f"Trained and saved new {kpi_type} model version {selected_version}")
                model_loaded[kpi_type] = True
            except Exception as e:
                with tab2:
                    st.error(f"Error training {kpi_type} model: {str(e)}")
                continue
    
    # Initialize model metrics
    metrics = {kpi_type: {'test': None, 'cv': None, 'full': None} for kpi_type in KPI_TYPES}
    predictions = {kpi_type: {'test': None, 'full': None} for kpi_type in KPI_TYPES}
    
    # Evaluate models
    for kpi_type in KPI_TYPES:
        if not model_loaded[kpi_type]:
            continue
        
        try:
            target = targets_dict[kpi_type]
            model = models[kpi_type]
            
            # Adapt features to match model's expected features
            if hasattr(model, 'feature_names_'):
                logger.info(f"Adapting features to match {kpi_type} model expectations")
                X_train_adapted = adapt_features_to_model(model, features)
                X_test_adapted, y_test_adapted, _, _ = train_test_split(X_train_adapted, target, test_size=0.2, random_state=42)
                
                # Use adapted features
                features_for_prediction = X_train_adapted
            else:
                X_test_adapted, y_test_adapted, _, _ = train_test_split(features, target, test_size=0.2, random_state=42)
                features_for_prediction = features
            
            # Cross-validation
            THRESHOLD = 0.5 if kpi_type == 'openrate' else 0.2 if kpi_type == 'clickrate' else 0.01
            sample_weights_train = np.where(target > THRESHOLD, 2.0, 1.0)
            cv_results = cross_validate_model(
                features_for_prediction, target, 
                kpi_type=kpi_type,
                params={'reg_lambda': 1.0, 'random_state': 42},
                sample_weights=sample_weights_train
            )
            
            # Test set evaluation
            test_metrics, y_pred_test = evaluate_model(model, X_test_adapted, y_test_adapted, kpi_type)
            
            # Full dataset verification
            full_metrics, y_pred_full = evaluate_model(model, features_for_prediction, target, kpi_type)
            
            # Store metrics and predictions
            metrics[kpi_type]['test'] = test_metrics
            metrics[kpi_type]['cv'] = cv_results
            metrics[kpi_type]['full'] = full_metrics
            predictions[kpi_type]['test'] = y_pred_test
            predictions[kpi_type]['full'] = y_pred_full
            
            # Generate documentation
            doc_dir = os.path.join(DOCS_DIR, f"{kpi_type}_model_v{selected_version}")
            if not os.path.exists(os.path.join(doc_dir, 'model_documentation.yaml')):
                generate_model_documentation(
                    model, feature_metadata, 
                    {'train_samples': features_for_prediction.shape[0]},
                    cv_results, test_metrics, 
                    kpi_type=kpi_type, version=selected_version
                )
                
                # Generate plots
                if hasattr(model, 'feature_names_'):
                    generate_feature_importance_plot(model, model.feature_names_, version=selected_version)
                else:
                    generate_feature_importance_plot(model, features.columns, version=selected_version)
                    
                generate_prediction_vs_actual_plot(y_test_adapted, y_pred_test, version=selected_version)
                generate_error_distribution_plot(y_test_adapted, y_pred_test, version=selected_version)
        except Exception as e:
            with tab2:
                st.error(f"Error evaluating {kpi_type} model: {str(e)}")
                logger.error(f"Error evaluating {kpi_type} model: {str(e)}")
                logger.error(traceback.format_exc())
            metrics[kpi_type]['test'] = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
            metrics[kpi_type]['cv'] = {'mse': {'mean': 0, 'std': 0}, 'rmse': {'mean': 0, 'std': 0}, 'mae': {'mean': 0, 'std': 0}, 'r2': {'mean': 0, 'std': 0}}
            metrics[kpi_type]['full'] = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
    
    # Tab 3: Feature Analysis
    with tab3:
        st.header('Feature Analysis')
        
        # Feature importance analysis
        st.subheader("Feature Importance Analysis")
        
        selected_kpi_for_analysis = st.selectbox(
            "Select KPI for Feature Analysis",
            options=KPI_TYPES,
            index=0
        )
        
        if model_loaded.get(selected_kpi_for_analysis, False):
            model = models[selected_kpi_for_analysis]
            
            try:
                importances = model.feature_importances_
                feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else features.columns
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top features
                top_n = st.slider("Number of top features to display", 5, 30, 15)
                top_features = importance_df.head(top_n)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(top_features['Feature'], top_features['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top {top_n} Feature Importances for {selected_kpi_for_analysis.capitalize()}')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature categories analysis
                st.subheader("Feature Categories Analysis")
                
                # Group features by type
                feature_categories = {
                    'Subject': [f for f in feature_names if 'Subject_' in f],
                    'Preheader': [f for f in feature_names if 'Preheader_' in f],
                    'Dialog': [f for f in feature_names if 'Dialog_' in f],
                    'Syfte': [f for f in feature_names if 'Syfte_' in f],
                    'Product': [f for f in feature_names if 'Product_' in f],
                    'Bolag': [f for f in feature_names if 'Bolag_' in f],
                    'Time': [f for f in feature_names if f in ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsMorning', 'IsAfternoon', 'IsEvening']],
                    'Age': [f for f in feature_names if f in ['Min_age', 'Max_age']]
                }
                
                # Calculate importance by category
                category_importance = {}
                for category, features_in_category in feature_categories.items():
                    matching_features = [f for f in features_in_category if f in importance_df['Feature'].values]
                    if matching_features:
                        total_importance = importance_df[importance_df['Feature'].isin(matching_features)]['Importance'].sum()
                        category_importance[category] = total_importance
                    else:
                        category_importance[category] = 0
                
                # Plot category importance
                category_df = pd.DataFrame({
                    'Category': list(category_importance.keys()),
                    'Importance': list(category_importance.values())
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(category_df['Category'], category_df['Importance'])
                ax.set_xlabel('Feature Category')
                ax.set_ylabel('Total Importance')
                ax.set_title(f'Feature Category Importance for {selected_kpi_for_analysis.capitalize()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show detailed feature importance table
                with st.expander("Detailed Feature Importance Table", expanded=False):
                    st.dataframe(importance_df)
            except Exception as e:
                st.error(f"Error analyzing feature importance: {str(e)}")
                logger.error(f"Error analyzing feature importance: {str(e)}")
        else:
            st.warning(f"No model loaded for {selected_kpi_for_analysis}. Please train a model first.")
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        try:
            # Combine features with targets
            analysis_data = features.copy()
            for kpi_type in KPI_TYPES:
                analysis_data[kpi_type] = targets_dict[kpi_type].values
            
            # Select features and target for correlation analysis
            target_for_corr = st.selectbox("Select target for correlation analysis", options=KPI_TYPES)
            
            # Correlation with the selected target
            numeric_cols = analysis_data.select_dtypes(include=['float64', 'int64']).columns
            correlations = analysis_data[numeric_cols].corr()[target_for_corr].drop(KPI_TYPES)
            
            # Sort and display top correlations
            corr_df = pd.DataFrame({
                'Feature': correlations.index,
                'Correlation': correlations.values
            }).sort_values('Correlation', key=abs, ascending=False)
            
            top_n_corr = st.slider("Number of top correlations to display", 5, 30, 15, key="corr_slider")
            top_corr = corr_df.head(top_n_corr)
            
            # Plot correlations
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(top_corr['Feature'], top_corr['Correlation'])
            
            # Color positive and negative correlations differently
            for i, bar in enumerate(bars):
                if top_corr['Correlation'].iloc[i] < 0:
                    bar.set_color('r')
                else:
                    bar.set_color('g')
                    
            ax.set_xlabel('Correlation')
            ax.set_title(f'Top {top_n_corr} Feature Correlations with {target_for_corr.capitalize()}')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed correlation table
            with st.expander("Detailed Correlation Table", expanded=False):
                st.dataframe(corr_df)
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
            logger.error(f"Error in correlation analysis: {str(e)}")
        
        # Text feature analysis
        st.subheader("Text Feature Analysis")
        
        try:
            # Calculate average KPIs by text feature
            text_features = {
                'Subject has exclamation mark': 'Subject_has_exclamation',
                'Subject has question mark': 'Subject_has_question',
                'Subject has numbers': 'Subject_has_numbers' if 'Subject_has_numbers' in features.columns else None,
                'Subject has currency symbol': 'Subject_has_currency' if 'Subject_has_currency' in features.columns else None,
                'Preheader has exclamation mark': 'Preheader_has_exclamation' if 'Preheader_has_exclamation' in features.columns else None,
                'Preheader has question mark': 'Preheader_has_question' if 'Preheader_has_question' in features.columns else None
            }
            
            # Filter out None values
            text_features = {k: v for k, v in text_features.items() if v is not None and v in features.columns}
            
            if text_features:
                selected_text_feature = st.selectbox(
                    "Select text feature for analysis",
                    options=list(text_features.keys())
                )
                
                feature_col = text_features[selected_text_feature]
                
                # Combine feature with targets
                analysis_df = pd.DataFrame({
                    'Feature': features[feature_col]
                })
                
                for kpi_type in KPI_TYPES:
                    analysis_df[kpi_type] = targets_dict[kpi_type].values
                
                # Group by feature and calculate average KPIs
                grouped = analysis_df.groupby('Feature').agg({kpi: 'mean' for kpi in KPI_TYPES}).reset_index()
                
                # Plot KPIs by feature value
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bar_width = 0.25
                index = np.arange(len(grouped['Feature']))
                
                for i, kpi in enumerate(KPI_TYPES):
                    ax.bar(index + i*bar_width, grouped[kpi], bar_width, label=kpi.capitalize())
                
                ax.set_xlabel('Feature Value (0 = No, 1 = Yes)')
                ax.set_ylabel('Average KPI Value')
                ax.set_title(f'Average KPIs by {selected_text_feature}')
                ax.set_xticks(index + bar_width)
                ax.set_xticklabels(grouped['Feature'])
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                st.dataframe(grouped)
            else:
                st.info("No suitable text features found for analysis. Try including NLP features in the sidebar.")
        except Exception as e:
            st.error(f"Error in text feature analysis: {str(e)}")
            logger.error(f"Error in text feature analysis: {str(e)}")
    
    # Tab 2: Model Results
    with tab2:
        st.header('Model Performance')
        
        # Model results tabs for each KPI
        kpi_tabs = st.tabs([kpi.capitalize() for kpi in KPI_TYPES])
        
        for i, kpi_type in enumerate(KPI_TYPES):
            with kpi_tabs[i]:
                st.subheader(f"{kpi_type.capitalize()} Model (Version: {selected_version})")
                
                if not model_loaded.get(kpi_type, False):
                    st.warning(f"No {kpi_type} model loaded. Please train a model first.")
                    continue
                
                # Model Parameter Information
                with st.expander("Model Parameters", expanded=False):
                    metadata = load_model_metadata(kpi_type, selected_version)
                    if metadata and 'model_parameters' in metadata:
                        params = metadata['model_parameters']
                        st.json(params)
                        
                        if 'sample_weights' in metadata:
                            st.subheader("Sample Weight Configuration")
                            st.json(metadata['sample_weights'])
                    else:
                        st.write("Model parameters not available in metadata.")
                        st.write(f"Current {kpi_type} XGBoost parameters:")
                        st.write(models[kpi_type].get_params())
                
                # Cross-validation results
                if metrics[kpi_type]['cv']:
                    st.subheader("Cross-Validation Performance (5-fold on Training Set)")
                    col1, col2 = st.columns(2)
                    col1.metric("Average Mean Squared Error", f"{metrics[kpi_type]['cv']['mse']['mean']:.6f}")
                    col1.metric("Average Root MSE", f"{metrics[kpi_type]['cv']['rmse']['mean']:.6f}")
                    col2.metric("Average Mean Absolute Error", f"{metrics[kpi_type]['cv']['mae']['mean']:.6f}")
                    col2.metric("Average R² Score", f"{metrics[kpi_type]['cv']['r2']['mean']:.4f}")
                
                # Test set results
                if metrics[kpi_type]['test']:
                    st.subheader("Test Set Performance")
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Squared Error", f"{metrics[kpi_type]['test']['mse']:.6f}")
                    col1.metric("Root MSE", f"{metrics[kpi_type]['test']['rmse']:.6f}")
                    col2.metric("Mean Absolute Error", f"{metrics[kpi_type]['test']['mae']:.6f}")
                    col2.metric("R² Score", f"{metrics[kpi_type]['test']['r2']:.4f}")
                
                # Full dataset verification
                if metrics[kpi_type]['full']:
                    st.subheader("Full Dataset Verification")
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Squared Error", f"{metrics[kpi_type]['full']['mse']:.6f}")
                    col1.metric("Root MSE", f"{metrics[kpi_type]['full']['rmse']:.6f}")
                    col2.metric("Mean Absolute Error", f"{metrics[kpi_type]['full']['mae']:.6f}")
                    col2.metric("R² Score", f"{metrics[kpi_type]['full']['r2']:.4f}")
                
                # Feature importances
                st.subheader("Feature Importances")
                
                try:
                    model = models[kpi_type]
                    importances = model.feature_importances_
                    feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else features.columns
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    top_features = importance_df.head(15)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(top_features['Feature'], top_features['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Top 15 Feature Importances for {kpi_type.capitalize()}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    with st.expander("All Feature Importances", expanded=False):
                        st.dataframe(importance_df)
                except Exception as e:
                    st.error(f"Error displaying feature importances: {str(e)}")
                
                # Predictions vs Actual
                if predictions[kpi_type]['test'] is not None:
                    st.subheader("Predictions vs Actual Values (Test Set)")
                    try:
                        # Get test data
                        X_test, y_test, _, _ = train_test_split(features, targets_dict[kpi_type], test_size=0.2, random_state=42)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_test, predictions[kpi_type]['test'], alpha=0.5)
                        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                        ax.set_xlabel(f'Actual {kpi_type.capitalize()}')
                        ax.set_ylabel(f'Predicted {kpi_type.capitalize()}')
                        ax.set_title(f'Actual vs Predicted {kpi_type.capitalize()} (Test Set)')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error displaying predictions vs actual plot: {str(e)}")
                
                # Error distribution
                if predictions[kpi_type]['test'] is not None:
                    st.subheader("Distribution of Prediction Errors (Test Set)")
                    try:
                        # Get test data
                        X_test, y_test, _, _ = train_test_split(features, targets_dict[kpi_type], test_size=0.2, random_state=42)
                        
                        errors = y_test - predictions[kpi_type]['test']
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(errors, bins=50)
                        ax.set_xlabel('Prediction Error')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {kpi_type.capitalize()} Prediction Errors (Test Set)')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error displaying error distribution plot: {str(e)}")
                
                # Hyperparameter tuning section
                with st.expander("Hyperparameter Tuning", expanded=False):
                    st.write("Tune hyperparameters to improve model performance.")
                    
                    if st.button(f"Tune {kpi_type.capitalize()} Model Hyperparameters", key=f"tune_{kpi_type}"):
                        with st.spinner(f"Tuning {kpi_type} model hyperparameters... This may take some time."):
                            try:
                                target = targets_dict[kpi_type]
                                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                                
                                # Configure sample weights
                                THRESHOLD = 0.5 if kpi_type == 'openrate' else 0.2 if kpi_type == 'clickrate' else 0.01
                                sample_weights_train = np.where(y_train > THRESHOLD, 2.0, 1.0)
                                
                                # Use reduced parameter grid for faster tuning in UI
                                param_grid = {
                                    'n_estimators': [100, 200],
                                    'max_depth': [4, 6],
                                    'learning_rate': [0.05, 0.1],
                                    'reg_lambda': [0.8, 1.2]
                                }
                                
                                best_params, best_score = tune_hyperparameters(
                                    X_train, y_train, 
                                    kpi_type=kpi_type,
                                    param_grid=param_grid,
                                    cv=3,  # Use fewer CV folds for speed
                                    sample_weights=sample_weights_train
                                )
                                
                                st.success(f"Hyperparameter tuning complete for {kpi_type} model!")
                                st.write("Best parameters:")
                                st.json(best_params)
                                st.write(f"Best RMSE score: {-best_score:.6f}")
                                
                                # Offer option to train with these parameters
                                if st.button(f"Train {kpi_type.capitalize()} Model with Best Parameters", key=f"train_best_{kpi_type}"):
                                    with st.spinner(f"Training {kpi_type} model with best parameters..."):
                                        # Train new model with best parameters
                                        new_model = train_model_with_params(
                                            X_train, y_train, 
                                            kpi_type=kpi_type,
                                            params=best_params,
                                            sample_weight_config={
                                                'threshold': THRESHOLD,
                                                'weight_high': 2.0,
                                                'weight_low': 1.0
                                            }
                                        )
                                        
                                        # Evaluate model
                                        test_metrics, y_pred_test = evaluate_model(new_model, X_test, y_test, kpi_type)
                                        
                                        # Save model
                                        model_file = get_model_filename(kpi_type, selected_version)
                                        joblib.dump(new_model, model_file)
                                        logger.info(f"Saved {kpi_type} model with best parameters to {model_file}")
                                        
                                        # Save metadata
                                        metadata = {
                                            'kpi_type': kpi_type,
                                            'version': selected_version,
                                            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'training_samples': X_train.shape[0],
                                            'test_samples': X_test.shape[0],
                                            'feature_names': X_train.columns.tolist(),
                                            'feature_count': X_train.shape[1],
                                            'include_preheader': include_preheader,
                                            'include_nlp': include_nlp,
                                            'include_time': include_time,
                                            'model_parameters': best_params,
                                            'sample_weights': {
                                                'threshold': THRESHOLD,
                                                'weight_high': 2.0,
                                                'weight_low': 1.0
                                            }
                                        }
                                        save_model_metadata(metadata, kpi_type, selected_version)
                                        
                                        # Update model in memory
                                        models[kpi_type] = new_model
                                        
                                        st.success(f"Successfully trained and saved {kpi_type} model with best parameters")
                                        st.info(f"Test set R² score: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.6f}")
                                        st.info("Refresh the page to see updated model results")
                            except Exception as e:
                                st.error(f"Error during hyperparameter tuning: {str(e)}")
                                logger.error(f"Error during hyperparameter tuning: {str(e)}")
                                logger.error(traceback.format_exc())
                                
if __name__ == '__main__':
    main()