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

def get_model_filename(version: str = None) -> str:
    """
    Generate model filename based on version
    """
    if version is None:
        version = get_current_model_version()
    return os.path.join(MODEL_BASE_DIR, f"xgboost_model_v{version}.pkl")

def get_model_metadata_filename(version: str = None) -> str:
    """
    Generate metadata filename based on version
    """
    if version is None:
        version = get_current_model_version()
    return os.path.join(MODEL_BASE_DIR, f"xgboost_model_v{version}_metadata.yaml")

def save_model_metadata(metadata: Dict[str, Any], version: str = None) -> None:
    """
    Save model metadata to a YAML file
    """
    try:
        if version is None:
            version = get_current_model_version()
        metadata_file = get_model_metadata_filename(version)
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"Model metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {str(e)}")
        logger.error(traceback.format_exc())

def load_model_metadata(version: str = None) -> Optional[Dict[str, Any]]:
    """
    Load model metadata from a YAML file
    """
    try:
        if version is None:
            version = get_current_model_version()
        metadata_file = get_model_metadata_filename(version)
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

def list_available_models() -> List[str]:
    """
    List all available model versions
    """
    try:
        model_files = [f for f in os.listdir(MODEL_BASE_DIR) if f.startswith('xgboost_model_v') and f.endswith('.pkl')]
        versions = [f.split('_v')[1].replace('.pkl', '') for f in model_files]
        versions.sort(key=lambda s: [int(u) for u in s.split('.')])
        return versions
    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        logger.error(traceback.format_exc())
        return []

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
def send_to_groq_api(subject_line: str, preheader: str, openrate_A: float,
                     selected_dialog: str, selected_syfte: str, selected_product: str,
                     min_age: int, max_age: int, included_bolag: List[str],
                     api_key: str = None) -> Dict[str, Any]:
    """
    Send data to Groq API for subject line and preheader suggestions
    """
    try:
        if api_key is None:
            # Try to get from environment
            api_key = os.getenv('GROQ_API_KEY')
            
        if not api_key:
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
                'Authorization': f'Bearer {api_key}',
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
        if 'Predicted_Openrate' in export_df.columns:
            export_df['Predicted_Openrate_Pct'] = export_df['Predicted_Openrate'].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        
        # Convert to CSV
        import io
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, encoding='utf-8', sep=';')
        
        return csv_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise

# --- Batch Prediction ---
def batch_predict(model, subjects: List[str], preheaders: List[str] = None, dialog: str = None, 
                  syfte: str = None, product: str = None, min_age: int = 18, max_age: int = 100, 
                  bolag: List[str] = None, include_preheader: bool = True) -> pd.DataFrame:
    """
    Perform batch prediction on multiple subject lines
    """
    # Validate inputs
    n_subjects = len(subjects)
    
    if preheaders is None:
        preheaders = [''] * n_subjects
    elif len(preheaders) != n_subjects:
        raise ValueError("Number of preheaders must match number of subjects")
    
    # Create batch data
    batch_data = pd.DataFrame({
        'Subject': subjects,
        'Preheader': preheaders,
        'Dialog': dialog or 'DEFAULT',
        'Syfte': syfte or 'DEFAULT',
        'Product': product or 'DEFAULT',
        'Min_age': min_age,
        'Max_age': max_age
    })
    
    # Create prediction features for each row
    predictions = []
    
    for idx, row in batch_data.iterrows():
        # Basic features
        features = {}
        
        # Subject features (both legacy and v2)
        subject = row['Subject']
        features['Subject_length'] = len(subject)
        features['Num_words'] = len(subject.split())
        features['Has_exclamation'] = 1 if '!' in subject else 0
        features['Has_question'] = 1 if '?' in subject else 0
        
        # v2 subject features
        features['Subject_num_words'] = len(subject.split())
        features['Subject_has_exclamation'] = 1 if '!' in subject else 0
        features['Subject_has_question'] = 1 if '?' in subject else 0
        features['Subject_caps_ratio'] = sum(1 for c in str(subject) if c.isupper()) / len(str(subject)) if len(str(subject)) > 0 else 0
        features['Subject_avg_word_len'] = np.mean([len(w) for w in str(subject).split()]) if len(str(subject).split()) > 0 else 0
        features['Subject_num_special_chars'] = sum(1 for c in str(subject) if c in '!?%$€£#@*&')
        features['Subject_first_word_len'] = len(str(subject).split()[0]) if len(str(subject).split()) > 0 else 0
        features['Subject_last_word_len'] = len(str(subject).split()[-1]) if len(str(subject).split()) > 0 else 0
        
        # Preheader features (v2 only)
        if include_preheader:
            preheader = row['Preheader']
            features['Preheader_length'] = len(preheader)
            features['Preheader_num_words'] = len(preheader.split())
            features['Preheader_has_exclamation'] = 1 if '!' in preheader else 0
            features['Preheader_has_question'] = 1 if '?' in preheader else 0
            features['Preheader_caps_ratio'] = sum(1 for c in str(preheader) if c.isupper()) / len(str(preheader)) if len(str(preheader)) > 0 else 0
            features['Preheader_avg_word_len'] = np.mean([len(w) for w in str(preheader).split()]) if len(str(preheader).split()) > 0 else 0
            features['Preheader_num_special_chars'] = sum(1 for c in str(preheader) if c in '!?%$€£#@*&')
            
            # Relationship features
            preheader_len = len(preheader) if len(preheader) > 0 else 1
            preheader_words = len(preheader.split()) if len(preheader.split()) > 0 else 1
            features['Subject_preheader_length_ratio'] = len(subject) / preheader_len
            features['Subject_preheader_words_ratio'] = len(subject.split()) / preheader_words
        
        # Age features
        features['Min_age'] = row['Min_age']
        features['Max_age'] = row['Max_age']
        
        # Try to predict
        try:
            # Check if model has feature information
            if hasattr(model, 'feature_names_'):
                # Create a DataFrame with all columns from the model
                input_data = pd.DataFrame(columns=model.feature_names_)
                input_data.loc[0] = 0
                
                # Set known features
                for feat_name, feat_value in features.items():
                    if feat_name in input_data.columns:
                        input_data[feat_name] = feat_value
                
                # Try to set categorical features
                dialog_col = f"Dialog_{row['Dialog']}"
                syfte_col = f"Syfte_{row['Syfte']}"
                product_col = f"Product_{row['Product']}"
                
                if dialog_col in input_data.columns:
                    input_data[dialog_col] = 1
                if syfte_col in input_data.columns:
                    input_data[syfte_col] = 1
                if product_col in input_data.columns:
                    input_data[product_col] = 1
                
                # Set bolag features if provided
                if bolag:
                    for b in bolag:
                        bolag_col = f'Bolag_{b}'
                        if bolag_col in input_data.columns:
                            input_data[bolag_col] = 1
                
                # Make prediction
                pred = model.predict(input_data)[0]
            else:
                # Model doesn't have feature information, so we can't predict accurately
                logger.warning("Model doesn't have feature_names_. Using simplified prediction.")
                pred = 0.5  # Default prediction
            
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error predicting for row {idx}: {str(e)}")
            predictions.append(None)
    
    # Add predictions to batch data
    batch_data['Predicted_Openrate'] = predictions
    
    return batch_data