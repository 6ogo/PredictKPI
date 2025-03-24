import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

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
        delivery_df = pd.read_csv("data/delivery_data.csv")
        customer_df = pd.read_csv("data/customer_data.csv")
        return delivery_df, customer_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def preprocess_data(delivery_df, customer_df):
    """Preprocess and merge the delivery and customer data."""
    if delivery_df is None or customer_df is None:
        return None
    
    # Clean delivery data
    delivery_df = delivery_df.copy()
    
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

def train_model(X, y, model_type='gradient_boosting'):
    """Train a regression model for predicting rates."""
    if X is None or y is None:
        return None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose the model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # Default to Ridge regression
        model = Ridge(alpha=1.0)
    
    # Train the model
    model.fit(X_train, y_train)
    
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

def prepare_features(df, text_vectorizers=None):
    """Prepare features for model training and prediction."""
    if df is None:
        return None
    
    # Define categorical and numerical features
    categorical_features = ['Dialog', 'Syfte', 'Produkt', 'MostCommonGender', 'MostCommonBolag', 
                           'Year', 'Month', 'DayOfWeek']
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
def main():
    st.title("Email Campaign Performance Predictor")
    
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
                # Create age bins
                df['AgeBin'] = pd.cut(df['AverageAge'], bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
                
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
        
        # Select target variable
        target = st.selectbox("Select Prediction Target", ["OpenRate", "ClickRate", "OptoutRate"])
        
        # Prepare features and target
        features_df = prepare_features(df, text_vectorizers)
        
        if features_df is None:
            st.error("Error preparing features for model training.")
            return
        
        # Select model type
        model_type = st.selectbox("Select Model Type", 
                                 ["gradient_boosting", "random_forest", "ridge"])
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, metrics, X_test, y_test, test_preds = train_model(
                    features_df, df[target], model_type=model_type
                )
                
                if model:
                    # Save model
                    model_filename = f"{target.lower()}_{model_type}_model.pkl"
                    vectorizers_filename = "text_vectorizers.pkl"
                    
                    if save_model(model, model_filename) and save_model(text_vectorizers, vectorizers_filename):
                        st.success(f"Model trained and saved as {model_filename}")
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    
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
                        plt.title('Top 15 Most Important Features')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Prediction page
    elif page == "Prediction":
        st.header("Predict Campaign Performance")
        
        # Check if models exist
        model_files = ['openrate_gradient_boosting_model.pkl', 
                      'clickrate_gradient_boosting_model.pkl', 
                      'optoutrate_gradient_boosting_model.pkl',
                      'text_vectorizers.pkl']
        
        models_exist = all(os.path.exists(f'models/{f}') for f in model_files)
        
        if not models_exist:
            st.warning("Models not found. Please train models on the 'Model Training' page first.")
            return
        
        # Load models
        open_model = load_model('openrate_gradient_boosting_model.pkl')
        click_model = load_model('clickrate_gradient_boosting_model.pkl')
        optout_model = load_model('optoutrate_gradient_boosting_model.pkl')
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
            
            with col1:
                gender = st.selectbox("Most Common Gender", ["Male", "Female", "Unknown"])
            
            with col2:
                age = st.number_input("Average Age", min_value=18, max_value=100, value=45)
            
            with col3:
                bolag = st.selectbox("Most Common Bolag", list(BOLAG_VALUES.keys()))
            
            # Number of unique customers
            unique_customers = st.number_input("Unique Customers", min_value=1, value=5000)
            
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