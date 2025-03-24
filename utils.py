"""
Utility functions for KPI Predictor application
"""
import os
import logging
import traceback
import datetime
import yaml
import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib

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

# Constants
MODEL_BASE_DIR = 'models'
DOCS_DIR = 'Docs'

# Ensure directories exist
for directory in [MODEL_BASE_DIR, DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Constants for Dropdown Options ---
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

# --- Version Management ---
def get_current_model_version() -> str:
    """
    Generate a model version based on the current date in YY.MM.DD format
    """
    today = datetime.datetime.now()
    return f"{today.strftime('%y.%m.%d')}"

def get_model_directory(version: str = None) -> str:
    """
    Get the directory for storing models of a specific version
    """
    if version is None:
        version = get_current_model_version()
    model_dir = os.path.join(MODEL_BASE_DIR, f"v{version}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def get_model_filename(kpi_type: str, version: str = None) -> str:
    """
    Generate model filename based on version and KPI type
    """
    model_dir = get_model_directory(version)
    return os.path.join(model_dir, f"{kpi_type}_model.pkl")

def get_model_metadata_filename(kpi_type: str, version: str = None) -> str:
    """
    Generate metadata filename based on version and KPI type
    """
    model_dir = get_model_directory(version)
    return os.path.join(model_dir, f"{kpi_type}_metadata.yaml")

def save_model_metadata(metadata: Dict[str, Any], kpi_type: str, version: str = None) -> None:
    """
    Save model metadata to a YAML file
    """
    try:
        metadata_file = get_model_metadata_filename(kpi_type, version)
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"Model metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {str(e)}")
        logger.error(traceback.format_exc())

def load_model_metadata(kpi_type: str, version: str = None) -> Optional[Dict[str, Any]]:
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
            logger.warning(f"Model metadata file not found: {metadata_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading model metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def list_available_model_versions() -> List[str]:
    """
    List all available model versions
    """
    try:
        model_dirs = [d for d in os.listdir(MODEL_BASE_DIR) if os.path.isdir(os.path.join(MODEL_BASE_DIR, d)) and d.startswith('v')]
        versions = [d[1:] for d in model_dirs]  # Remove the 'v' prefix
        versions.sort(key=lambda s: [int(u) for u in s.split('.')])
        return versions
    except Exception as e:
        logger.error(f"Error listing available model versions: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_available_kpi_models(version: str = None) -> Dict[str, List[str]]:
    """
    Get available KPI models for a specific version
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping version to list of available KPI types
    """
    try:
        result = {}
        
        if version:
            versions = [version]
        else:
            versions = list_available_model_versions()
            
        for ver in versions:
            model_dir = get_model_directory(ver)
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
                kpi_types = [f.split('_model.pkl')[0] for f in model_files]
                result[ver] = kpi_types
                
        return result
    except Exception as e:
        logger.error(f"Error getting available KPI models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# --- Age Group Utils ---
def categorize_age(age: int) -> str:
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

# --- API Interaction ---
def send_to_groq_api(subject_line: str, preheader: str, 
                    kpi_predictions: Dict[str, float],
                    campaign_metadata: Dict[str, Any],
                    api_key: str = None) -> Dict[str, Any]:
    """
    Send data to Groq API for subject line and preheader suggestions
    
    Parameters:
    -----------
    subject_line : str
        Current subject line
    preheader : str
        Current preheader
    kpi_predictions : Dict[str, float]
        Dictionary of KPI predictions for current content
    campaign_metadata : Dict[str, Any]
        Dictionary of campaign metadata
    api_key : str, optional
        Groq API key
        
    Returns:
    --------
    Dict[str, Any]
        API response with suggestions
    """
    try:
        if api_key is None:
            # Try to get from environment
            api_key = os.getenv('GROQ_API_KEY')
            
        if not api_key:
            return {"error": "Groq API key not found. Please set GROQ_API_KEY in .env file."}
        
        # Extract KPI predictions
        openrate = kpi_predictions.get('openrate', 0)
        clickrate = kpi_predictions.get('clickrate', 0)
        optoutrate = kpi_predictions.get('optoutrate', 0)
        
        # Extract campaign metadata
        dialog = campaign_metadata.get('dialog_display', '')
        syfte = campaign_metadata.get('syfte_display', '')
        product = campaign_metadata.get('product_display', '')
        min_age = campaign_metadata.get('min_age', 18)
        max_age = campaign_metadata.get('max_age', 100)
        included_bolag = campaign_metadata.get('included_bolag', [])
        
        prompt = f"""
        I need to create email subject lines and preheaders for a marketing campaign. 
        
        Current subject line: "{subject_line}"
        Current preheader: "{preheader}"
        Current predicted metrics:
        - Open rate: {openrate:.2%}
        - Click rate: {clickrate:.2%}
        - Opt-out rate: {optoutrate:.2%}
        
        Campaign details:
        - Dialog: {dialog}
        - Syfte (Purpose): {syfte}
        - Product: {product}
        - Age range: {min_age} to {max_age}
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
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are an expert in email marketing optimization. Your task is to generate compelling email subject lines and preheaders in Swedish that maximize open rates and click-through rates while minimizing opt-outs."},
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

# --- Data Loading ---
def load_and_prepare_data(delivery_file='Data/delivery_data.csv', customer_file='Data/customer_data.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data from CSV files
    
    Parameters:
    -----------
    delivery_file : str
        Path to delivery data CSV file
    customer_file : str
        Path to customer data CSV file
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Prepared delivery data and customer data
    """
    try:
        logger.info(f"Loading data from {delivery_file} and {customer_file}")
        
        # Load data
        delivery_data = pd.read_csv(delivery_file, sep=';', encoding='utf-8')
        customer_data = pd.read_csv(customer_file, sep=';', encoding='utf-8')
        
        # Validate required columns
        delivery_required_cols = ['InternalName', 'Subject', 'Preheader', 'Date', 'Sendouts', 'Opens', 'Clicks', 'Optouts', 'Dialog', 'Syfte', 'Product']
        customer_required_cols = ['Primary key', 'InternalName', 'OptOut', 'Open', 'Click', 'Gender', 'Age', 'Bolag']
        
        delivery_missing = [col for col in delivery_required_cols if col not in delivery_data.columns]
        customer_missing = [col for col in customer_required_cols if col not in customer_data.columns]
        
        if delivery_missing:
            logger.warning(f"Missing delivery columns: {delivery_missing}")
            
            # Add missing columns with default values
            for col in delivery_missing:
                if col == 'Date':
                    delivery_data[col] = pd.Timestamp('now')
                elif col == 'Preheader':
                    delivery_data[col] = ''
                else:
                    delivery_data[col] = np.nan
        
        if customer_missing:
            logger.warning(f"Missing customer columns: {customer_missing}")
            
            # Add missing columns with default values
            for col in customer_missing:
                if col in ['OptOut', 'Open', 'Click']:
                    customer_data[col] = 0
                elif col == 'Age':
                    customer_data[col] = 30  # Default age
                elif col == 'Gender':
                    customer_data[col] = 'Unknown'
                elif col == 'Bolag':
                    customer_data[col] = 'Unknown'
                else:
                    customer_data[col] = np.nan
        
        # Calculate KPIs
        delivery_data['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
        delivery_data['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
        delivery_data['Optoutrate'] = delivery_data['Optouts'] / delivery_data['Sendouts']
        
        # Fix data types
        if 'Date' in delivery_data.columns:
            delivery_data['Date'] = pd.to_datetime(delivery_data['Date'], errors='coerce')
            
        if 'Age' in customer_data.columns:
            customer_data['Age'] = pd.to_numeric(customer_data['Age'], errors='coerce')
            
        # Fill missing values
        for col in ['Subject', 'Preheader']:
            if col in delivery_data.columns:
                delivery_data[col] = delivery_data[col].fillna('')
        
        for col in ['OptOut', 'Open', 'Click']:
            if col in customer_data.columns:
                customer_data[col] = customer_data[col].fillna(0).astype(int)
        
        return delivery_data, customer_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
# --- CSV Import/Export ---
def import_subjects_from_csv(file) -> pd.DataFrame:
    """
    Import subject lines and optionally preheaders from CSV file
    """
    try:
        # Read the file
        if hasattr(file, 'getvalue'):
            # Streamlit's UploadedFile 
            content = file.getvalue().decode('utf-8')
        else:
            # Direct file content
            content = file.read().decode('utf-8')
        
        # Try different delimiters
        import io
        sniffer = pd.read_csv(io.StringIO(content), nrows=5)
        if len(sniffer.columns) <= 1:
            # Try semicolon delimiter
            df = pd.read_csv(io.StringIO(content), sep=';')
        else:
            # Standard CSV
            df = pd.read_csv(io.StringIO(content))
        
        # Check for required columns
        required_cols = ['Subject']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error importing CSV: {str(e)}")
        raise

def export_results_to_csv(results: pd.DataFrame) -> str:
    """
    Export prediction results to CSV
    
    Returns:
    --------
    str: CSV content as string
    """
    try:
        # Format columns for export
        export_df = results.copy()
        
        # Format rates as percentages for display
        rate_columns = ['Predicted_Openrate', 'Predicted_Clickrate', 'Predicted_Optoutrate']
        for col in rate_columns:
            if col in export_df.columns:
                export_df[f"{col}_Pct"] = export_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        
        # Convert to CSV
        import io
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, encoding='utf-8', sep=';')
        
        return csv_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise