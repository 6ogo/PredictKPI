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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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

CURRENT_MODEL_VERSION = get_current_model_version()  # e.g., 25.03.18 for March 18, 2025

# --- Enum Constants ---
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

# Ensure directories exist
for directory in [MODEL_BASE_DIR, DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

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

def create_age_heatmap(heatmap_data, metric, title, cmap='viridis', figsize=(12, 6)):
    """
    Create a heatmap for a specific metric by age group
    """
    try:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data for this metric
        data = heatmap_data[metric].copy()
        
        # Ensure data is numeric
        data = data.astype(float)
        
        # Format values as percentages
        labels = data.applymap(lambda x: f"{x:.2%}")
        
        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                ax.text(j, i, labels.iloc[i, j],
                        ha="center", va="center", 
                        color="white" if data.iloc[i, j] > data.values.mean() else "black")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f"{title} (%)", rotation=-90, va="bottom")
        
        # Add title and adjust layout
        ax.set_title(title)
        fig.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_interactive_heatmap(data, metric, title, is_percentage=True, colorscale='Viridis'):
    """
    Create an interactive heatmap using Plotly for a specific metric
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with age groups as index and categories as columns
    metric : str
        The metric name ('Openrate', 'Clickrate', or 'Optoutrate')
    title : str
        The title for the heatmap
    is_percentage : bool
        Whether to format values as percentages
    colorscale : str
        Plotly colorscale to use
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    
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
    pd.DataFrame
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

# --- Model Version Management ---
def get_model_filename(version=CURRENT_MODEL_VERSION):
    """
    Generate model filename based on version
    """
    return os.path.join(MODEL_BASE_DIR, f"xgboost_model_v{version}.pkl")

def get_model_metadata_filename(version=CURRENT_MODEL_VERSION):
    """
    Generate metadata filename based on version
    """
    return os.path.join(MODEL_BASE_DIR, f"xgboost_model_v{version}_metadata.yaml")

def save_model_metadata(metadata, version=CURRENT_MODEL_VERSION):
    """
    Save model metadata to a YAML file
    """
    try:
        metadata_file = get_model_metadata_filename(version)
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"Model metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {str(e)}")
        logger.error(traceback.format_exc())

def load_model_metadata(version=CURRENT_MODEL_VERSION):
    """
    Load model metadata from a YAML file
    """
    try:
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

def list_available_models():
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

# --- Feature Engineering ---
def engineer_features(delivery_data, customer_data, include_preheader=True):
    """
    Perform feature engineering on the dataset
    
    Parameters:
    delivery_data (pd.DataFrame): Delivery data
    customer_data (pd.DataFrame): Customer data
    include_preheader (bool): Whether to include preheader features
    
    Returns:
    tuple: features DataFrame, target Series, and a dictionary of feature metadata
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
        
        # Legacy feature names (version 1.x)
        delivery_data['Subject_length'] = delivery_data['Subject'].str.len()
        delivery_data['Num_words'] = delivery_data['Subject'].str.split().str.len()
        delivery_data['Has_exclamation'] = delivery_data['Subject'].str.contains('!').astype(int)
        delivery_data['Has_question'] = delivery_data['Subject'].str.contains(r'\?', regex=True).astype(int)
        
        # Enhanced text features for subject (version 2.x+)
        delivery_data['Subject_num_words'] = delivery_data['Subject'].str.split().str.len()
        delivery_data['Subject_has_exclamation'] = delivery_data['Subject'].str.contains('!').astype(int)
        delivery_data['Subject_has_question'] = delivery_data['Subject'].str.contains(r'\?', regex=True).astype(int)
        
        # Add more sophisticated text features for subject
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
        
        # Add preheader features if requested (for version 2.x)
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
        
        # Define feature columns
        categorical_features = ['Dialog', 'Syfte', 'Product']
        
        # Define numerical features based on version
        legacy_numerical_features = [
            'Min_age', 'Max_age', 
            'Subject_length', 'Num_words', 'Has_exclamation', 'Has_question'
        ]
        
        v2_numerical_features = [
            'Min_age', 'Max_age', 
            'Subject_length', 'Subject_num_words', 'Subject_has_exclamation', 'Subject_has_question',
            'Subject_caps_ratio', 'Subject_avg_word_len', 'Subject_num_special_chars',
            'Subject_first_word_len', 'Subject_last_word_len'
        ]
        
        if include_preheader:
            v2_numerical_features.extend([
                'Preheader_length', 'Preheader_num_words', 
                'Preheader_has_exclamation', 'Preheader_has_question',
                'Preheader_caps_ratio', 'Preheader_avg_word_len', 'Preheader_num_special_chars',
                'Subject_preheader_length_ratio', 'Subject_preheader_words_ratio'
            ])
        
        bolag_features_list = [col for col in delivery_data.columns if col.startswith('Bolag_')]
        
        # Track feature sets for different model versions
        feature_sets = {
            'legacy': {
                'categorical': categorical_features,
                'numerical': legacy_numerical_features,
                'bolag': bolag_features_list
            },
            'v2': {
                'categorical': categorical_features,
                'numerical': v2_numerical_features,
                'bolag': bolag_features_list
            }
        }
        
        # Generate categorical dummies
        dummy_df = pd.get_dummies(delivery_data[categorical_features])
        
        # Create mappings for UI
        dummy_dialog_map = {dialog: f'Dialog_{dialog}' for dialog in delivery_data['Dialog'].unique()}
        dummy_syfte_map = {syfte: f'Syfte_{syfte}' for syfte in delivery_data['Syfte'].unique()}
        dummy_product_map = {product: f'Product_{product}' for product in delivery_data['Product'].unique()}
        
        # Prepare features for legacy model (v1.x)
        legacy_features = pd.concat([
            dummy_df,
            delivery_data[legacy_numerical_features],
            delivery_data[bolag_features_list].fillna(0).astype(int)
        ], axis=1)
        
        # Prepare features for v2 model
        v2_features = pd.concat([
            dummy_df,
            delivery_data[v2_numerical_features],
            delivery_data[bolag_features_list].fillna(0).astype(int)
        ], axis=1)
        
        # Target variable
        target = delivery_data['Openrate']
        
        # Metadata for documentation
        feature_metadata = {
            'categorical_features': categorical_features,
            'legacy_numerical_features': legacy_numerical_features,
            'v2_numerical_features': v2_numerical_features,
            'bolag_features': bolag_features_list,
            'dummy_dialog_map': dummy_dialog_map,
            'dummy_syfte_map': dummy_syfte_map,
            'dummy_product_map': dummy_product_map,
            'feature_sets': feature_sets,
            'include_preheader': include_preheader
        }
        
        logger.info(f"Feature engineering completed - Legacy features: {legacy_features.shape}, V2 features: {v2_features.shape}")
        
        return {
            'legacy': legacy_features, 
            'v2': v2_features
        }, target, feature_metadata
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error in feature engineering: {str(e)}")
        raise

# --- Model Training and Validation ---
def train_model(X_train, y_train, sample_weights=None, params=None):
    """
    Train an XGBoost model with the given parameters
    """
    try:
        if params is None:
            params = {
                'reg_lambda': 1.0,
                'random_state': 42
            }
        
        logger.info(f"Training XGBoost model with parameters: {params}")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model_with_params(X_train, y_train, params, sample_weight_config=None):
    """
    Train an XGBoost model with the given parameters and sample weight configuration
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable
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
        logger.info(f"Training XGBoost model with parameters: {params}")
        
        # Configure sample weights if provided
        sample_weights = None
        if sample_weight_config is not None:
            threshold = sample_weight_config.get('threshold', 0.5)
            weight_high = sample_weight_config.get('weight_high', 2.0)
            weight_low = sample_weight_config.get('weight_low', 1.0)
            
            logger.info(f"Using sample weights - threshold: {threshold}, high: {weight_high}, low: {weight_low}")
            sample_weights = np.where(y_train > threshold, weight_high, weight_low)
        
        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return model
    except Exception as e:
        logger.error(f"Error training model with parameters: {str(e)}")
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
def evaluate_model(model, X_test, y_test):
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
        
        return metrics, y_pred
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cross_validate_model(X_train, y_train, params=None, n_splits=5, sample_weights=None):
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
        
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
    """
    try:
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
        st.info("Tuning hyperparameters... This may take some time.")
        
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
def generate_model_documentation(model, feature_metadata, train_metrics, cv_results, test_metrics, version=CURRENT_MODEL_VERSION):
    """
    Generate model documentation
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create documentation directory
        model_doc_dir = os.path.join(DOCS_DIR, f"model_v{version}")
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

def generate_feature_importance_plot(model, feature_names, version=CURRENT_MODEL_VERSION):
    """
    Generate and save feature importance plot
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join(DOCS_DIR, f"model_v{version}")
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

def generate_prediction_vs_actual_plot(y_test, y_pred, version=CURRENT_MODEL_VERSION):
    """
    Generate and save prediction vs actual plot
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join(DOCS_DIR, f"model_v{version}")
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

def generate_error_distribution_plot(y_test, y_pred, version=CURRENT_MODEL_VERSION):
    """
    Generate and save error distribution plot
    """
    try:
        # Create documentation directory
        model_doc_dir = os.path.join(DOCS_DIR, f"model_v{version}")
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

# --- API Interaction ---
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
    st.title('Sendout KPI Predictor')
    
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
        selected_version = CURRENT_MODEL_VERSION
        st.sidebar.info(f"No saved models found. Will train version {CURRENT_MODEL_VERSION} if needed.")
    
    model_file = get_model_filename(selected_version)
    metadata_file = get_model_metadata_filename(selected_version)
    
    # Force retrain option
    force_retrain = st.sidebar.checkbox("Force model retraining")
    
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
        features_dict, target, feature_metadata = engineer_features(
            delivery_data, customer_data, 
            include_preheader=include_preheader
        )
    except Exception as e:
        st.error(f"Failed to engineer features: {str(e)}")
        return
    
    # Select appropriate feature set based on version
    if float(selected_version.split('.')[0]) >= 2:
        features = features_dict['v2']
        feature_set_key = 'v2'
    else:
        features = features_dict['legacy']
        feature_set_key = 'legacy'
    
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
            model = joblib.load(model_file)
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
            model = train_model(X_train, y_train, sample_weights=sample_weights_train, params=params)
            
            # Save model
            joblib.dump(model, model_file)
            logger.info(f"Saved model to {model_file}")
            
            # Save metadata
            metadata = {
                'version': selected_version,
                'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'feature_set': feature_set_key,
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_names': X_train.columns.tolist(),
                'feature_count': X_train.shape[1],
                'include_preheader': include_preheader,
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
        
        # Cross-validation
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
        if not os.path.exists(os.path.join(DOCS_DIR, f"model_v{selected_version}", 'model_documentation.yaml')):
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
    tab1, tab2 = st.tabs(['Sendout Prediction', 'Model Results'])
    
    # Tab 2: Model Performance
    with tab2:
        st.header('Model Performance')
        st.subheader(f"Model Version: {selected_version}")
        
        # Add retraining section with parameters
        with st.expander("Retrain Model with Custom Parameters", expanded=False):
            st.write("Adjust model parameters and click 'Retrain Model' to create a new version.")
            
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
                    sw_threshold = 0.5
                    sw_weight_high = 1.0
                    sw_weight_low = 1.0
            
            # Version options
            st.subheader("Version Control")
            use_today_version = st.checkbox("Use today's date for version", value=True, 
                                            help="Creates a version based on today's date (YY.MM.DD)")
            if not use_today_version:
                custom_version = st.text_input("Custom Version", value=selected_version, 
                                            help="Specify a custom version number (e.g., '2.1.0')")
            
            # Retrain button
            if st.button("Retrain Model"):
                try:
                    # Set version
                    new_version = get_current_model_version() if use_today_version else custom_version
                    
                    # Show training status
                    with st.spinner(f"Training model version {new_version}..."):
                        # Configure parameters
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
                            sample_weight_config=sample_weight_config
                        )
                        
                        # Save model
                        new_model_file = get_model_filename(new_version)
                        joblib.dump(new_model, new_model_file)
                        
                        # Evaluate new model
                        test_metrics, y_pred_test = evaluate_model(new_model, X_test, y_test)
                        
                        # Save metadata
                        metadata = {
                            'version': new_version,
                            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'feature_set': feature_set_key,
                            'training_samples': X_train.shape[0],
                            'test_samples': X_test.shape[0],
                            'feature_names': X_train.columns.tolist(),
                            'feature_count': X_train.shape[1],
                            'include_preheader': include_preheader,
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
                        
                        if hasattr(new_model, 'feature_names_'):
                            generate_feature_importance_plot(new_model, new_model.feature_names_, version=new_version)
                        else:
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
                    st.write("Current XGBoost parameters:")
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
                        fig = create_age_heatmap(filtered_data, 'Openrate', 'Open Rate by Age Group')
                        st.pyplot(fig)
                    
                    with metric_tabs[1]:
                        fig = create_age_heatmap(filtered_data, 'Clickrate', 'Click Rate by Age Group')
                        st.pyplot(fig)
                    
                    with metric_tabs[2]:
                        fig = create_age_heatmap(filtered_data, 'Optoutrate', 'Opt-out Rate by Age Group')
                        st.pyplot(fig)
                    
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
            ax.set_title('Top 15 Feature Importances')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("All Feature Importances")
            st.dataframe(importance_df)
        except Exception as e:
            st.error(f"Error displaying feature importances: {str(e)}")
        
        # Dataset information
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {features.shape[0]}")
        st.write(f"Number of features: {features.shape[1]}")
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual Values (Test Set)")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.5)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel('Actual Open Rate')
            ax.set_ylabel('Predicted Open Rate')
            ax.set_title('Actual vs Predicted Open Rates (Test Set)')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying predictions vs actual plot: {str(e)}")
        
        # Error distribution
        st.subheader("Distribution of Prediction Errors (Test Set)")
        try:
            errors = y_test - y_pred_test
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(errors, bins=50)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Prediction Errors (Test Set)')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying error distribution plot: {str(e)}")
        
        # Documentation links
        st.subheader("Model Documentation")
        doc_path = os.path.join(DOCS_DIR, f"model_v{selected_version}")
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
        
        # Subject line and Preheader input with GenAI checkbox
        col1, col2 = st.columns([3, 1])
        with col1:
            subject_line = st.text_input('Subject Line')
            # Only show preheader if using v2+ model
            if float(selected_version.split('.')[0]) >= 2:
                preheader = st.text_input('Preheader')
            else:
                preheader = ""
                st.info("Preheader not used in this model version")
        with col2:
            use_genai = st.checkbox('GenAI', value=False)
        
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
                                        subject = sug.get('subject', '')
                                        preheader_alt = sug.get('preheader', '')
                                        
                                        if subject and (preheader_alt or float(selected_version.split('.')[0]) < 2):
                                            openrate = predict_for_subject_and_preheader(subject, preheader_alt)
                                            options.append((chr(65 + i), subject, preheader_alt, openrate))
                                    
                                    if options:
                                        # Add current option as Version A
                                        all_options = [('A', subject_line, preheader, openrate_A)] + options
                                        
                                        st.subheader("A/B/C/D Test Results")
                                        
                                        # Create a container with a light background for all versions
                                        results_container = st.container()
                                        with results_container:
                                            # Display each version in its own box
                                            for opt, subject, preheader_text, openrate in all_options:
                                                # Create a container with border for each version
                                                with st.expander(f"**Version {opt}**", expanded=True):
                                                    is_current = opt == 'A'
                                                    is_best = openrate == max(o[3] for o in all_options)
                                                    
                                                    # Add a "Current" or "Best" badge if applicable
                                                    badges = []
                                                    if is_current:
                                                        badges.append("🔹 Current")
                                                    if is_best:
                                                        badges.append("⭐ Best")
                                                    
                                                    if badges:
                                                        st.markdown(f"<div style='margin-bottom:10px'>{' | '.join(badges)}</div>", unsafe_allow_html=True)
                                                    
                                                    # Subject and preheader
                                                    st.markdown("**Subject:**")
                                                    st.markdown(f"<div style='background-color:rgb(133, 133, 133);padding:10px;border-radius:5px;margin-bottom:10px'>{subject}</div>", unsafe_allow_html=True)
                                                    
                                                    if float(selected_version.split('.')[0]) >= 2:
                                                        st.markdown("**Preheader:**")
                                                        st.markdown(f"<div style='background-color:rgb(133, 133, 133);padding:10px;border-radius:5px;margin-bottom:10px'>{preheader_text}</div>", unsafe_allow_html=True)
                                                    
                                                    # Metrics
                                                    st.markdown("**Predicted Results:**")
                                                    col1, col2, col3 = st.columns(3)
                                                    delta = None if is_current else openrate - openrate_A
                                                    col1.metric("Open Rate", f"{openrate:.2%}", f"{delta:.2%}" if delta is not None else None)
                                                    col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                                                    col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                                        
                                        # Version comparison heatmaps - put this OUTSIDE the version expanders
                                        # to avoid nested expanders which cause the error
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
                                        
                                        # Find the best option
                                        best_option = max(all_options, key=lambda x: x[3])
                                        
                                        # Summary section
                                        st.subheader("Summary")
                                        st.write(f"Best performing version: **Version {best_option[0]}** with {best_option[3]:.2%} predicted open rate")
                                        
                                        if best_option[0] != 'A':
                                            improvement = best_option[3] - openrate_A
                                            st.write(f"Improvement over current version: **{improvement:.2%}**")
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