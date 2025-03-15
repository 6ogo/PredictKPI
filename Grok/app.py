import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import requests
import json
from dotenv import load_dotenv
import os
import joblib

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Data Loading and Preprocessing ---
delivery_data = pd.read_csv('../Data/delivery_data.csv', sep=';', encoding='utf-8')
customer_data = pd.read_csv('../Data/customer_data.csv', sep=';', encoding='utf-8')

# Display some debug information
print("Delivery data shape:", delivery_data.shape)
print("Customer data shape:", customer_data.shape)
print("Delivery data columns:", delivery_data.columns.tolist())
print("Customer data columns:", customer_data.columns.tolist())

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

# Extract subject line features
delivery_data['Subject_length'] = delivery_data['Subject'].str.len()
delivery_data['Num_words'] = delivery_data['Subject'].str.split().str.len()
delivery_data['Has_exclamation'] = delivery_data['Subject'].str.contains('!').astype(int)
delivery_data['Has_question'] = delivery_data['Subject'].str.contains(r'\?', regex=True).astype(int)

# Define feature columns
categorical_features = ['Dialog', 'Syfte', 'Product']
numerical_features = ['Min_age', 'Max_age', 'Subject_length', 'Num_words', 'Has_exclamation', 'Has_question']
bolag_features_list = [col for col in delivery_data.columns if col.startswith('Bolag_')]

# Prepare features and target
features = pd.concat([
    pd.get_dummies(delivery_data[categorical_features]),
    delivery_data[numerical_features],
    delivery_data[bolag_features_list].fillna(0).astype(int)  # Fill NaN with 0 then convert to int
], axis=1)
target = delivery_data['Openrate']

# Checking syfte, dialog, and product columns
dummy_df = pd.get_dummies(delivery_data[categorical_features])
print("Dummy columns:", dummy_df.columns.tolist())

# Explicitly store the mapping from actual data to dummies columns
dummy_dialog_map = {}
for dialog in delivery_data['Dialog'].unique():
    dummy_dialog_map[dialog] = f'Dialog_{dialog}'

dummy_syfte_map = {}
for syfte in delivery_data['Syfte'].unique():
    dummy_syfte_map[syfte] = f'Syfte_{syfte}'
    
dummy_product_map = {}
for product in delivery_data['Product'].unique():
    dummy_product_map[product] = f'Product_{product}'

print("Dialog mapping:", dummy_dialog_map)
print("Syfte mapping:", dummy_syfte_map)
print("Product mapping:", dummy_product_map)

# Load or train XGBoost model
model_file = 'xgboost_model.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
    # Initialize mse with a default value since we're loading a pre-trained model
    mse = 0.0
else:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

# --- Define Enums ---
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

# Filter enums based on data
available_bolag = customer_data['Bolag'].unique().tolist()
available_syfte = delivery_data['Syfte'].unique().tolist()
available_dialog = delivery_data['Dialog'].unique().tolist()
available_produkt = delivery_data['Product'].unique().tolist()

# Debug prints to see what's available
print("Available Bolag:", available_bolag)
print("Available Syfte:", available_syfte)
print("Available Dialog:", available_dialog)
print("Available Product:", available_produkt)

# Check bolag mapping
bolag_options = []
for b in available_bolag:
    for key, value in BOLAG_VALUES.items():
        if value == b:
            bolag_options.append(key)
            break

# --- Streamlit App ---
st.title('Sendout KPI Predictor')

# Create tabs
tab1, tab2 = st.tabs(['Sendout Prediction', 'Model Results'])

# Tab 2: Model Performance
with tab2:
    st.header('Model Performance')
    
    # Check if the model was loaded or trained
    if os.path.exists(model_file):
        st.info("Model was loaded from saved file. Let's verify its performance on the dataset.")
        
        # Evaluate model on the whole dataset to verify performance
        X_verify = features
        y_verify = target
        y_pred = model.predict(X_verify)
        
        # Calculate metrics
        mse = mean_squared_error(y_verify, y_pred)
        mae = np.mean(np.abs(y_verify - y_pred))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_verify, y_pred)
        
        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.6f}")
        col1.metric("Root MSE", f"{rmse:.6f}")
        col2.metric("Mean Absolute Error", f"{mae:.6f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        # Display feature importances
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        feature_names = features.columns
        
        # Create DataFrame for better display
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Show top 15 features in a bar chart
        top_features = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_features['Feature'], top_features['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display the full feature importance table
        st.subheader("All Feature Importances")
        st.dataframe(importance_df)
        
        # Check data size
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {X_verify.shape[0]}")
        st.write(f"Number of features: {X_verify.shape[1]}")
        
        # Compare predictions vs actual values
        st.subheader("Predictions vs Actual Values")
        
        # Create a scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_verify, y_pred, alpha=0.5)
        ax.plot([min(y_verify), max(y_verify)], [min(y_verify), max(y_verify)], 'r--')
        ax.set_xlabel('Actual Open Rate')
        ax.set_ylabel('Predicted Open Rate')
        ax.set_title('Actual vs Predicted Open Rates')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribution of prediction errors
        errors = y_verify - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors')
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # For newly trained model
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        st.write(f'Training set size: {X_train.shape[0]} samples')
        st.write(f'Test set size: {X_test.shape[0]} samples')
        st.write(f'Mean Squared Error on test set: {mse:.6f}')
        st.write(f'Root Mean Squared Error: {np.sqrt(mse):.6f}')
        st.write(f'Mean Absolute Error: {np.mean(np.abs(y_test - model.predict(X_test))):.6f}')
        st.write(f'R² Score: {r2_score(y_test, model.predict(X_test)):.4f}')
        
        importances = model.feature_importances_
        feature_names = features.columns
        
        # Show top 15 features
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
        
        # Display full feature importance table
        st.dataframe(importance_df)

# Tab 1: Sendout Prediction
with tab1:
    st.header('Predict KPIs for New Sendout')

    # Show available dialogs directly from the data, not from the DIALOG_VALUES mapping
    dialog_options = sorted(delivery_data['Dialog'].unique().tolist())
    dialog_labels = []
    for d in dialog_options:
        # Try to find a friendly name in DIALOG_VALUES, otherwise use the code
        found = False
        for key, value in DIALOG_VALUES.items():
            if value[0] == d:
                dialog_labels.append((d, value[1]))
                found = True
                break
        if not found:
            dialog_labels.append((d, d))  # Use the code as the label
            
    selected_dialog = st.selectbox(
        'Dialog', 
        options=[label for _, label in dialog_labels],
        format_func=lambda x: x
    )
    # Map the selected display name back to the code
    for code, label in dialog_labels:
        if label == selected_dialog:
            selected_dialog_code = code
            break
    
    # Same for Syfte
    syfte_options = sorted(delivery_data['Syfte'].unique().tolist())
    syfte_labels = []
    for s in syfte_options:
        found = False
        for key, value in SYFTE_VALUES.items():
            if value[0] == s:
                syfte_labels.append((s, value[1]))
                found = True
                break
        if not found:
            syfte_labels.append((s, s))
            
    selected_syfte = st.selectbox(
        'Syfte', 
        options=[label for _, label in syfte_labels],
        format_func=lambda x: x
    )
    # Map back to code
    for code, label in syfte_labels:
        if label == selected_syfte:
            selected_syfte_code = code
            break
    
    # Same for Product
    product_options = sorted(delivery_data['Product'].unique().tolist())
    product_labels = []
    for p in product_options:
        found = False
        for key, value in PRODUKT_VALUES.items():
            if value[0] == p:
                product_labels.append((p, value[1]))
                found = True
                break
        if not found:
            product_labels.append((p, p))
            
    selected_product = st.selectbox(
        'Product', 
        options=[label for _, label in product_labels],
        format_func=lambda x: x
    )
    # Map back to code
    for code, label in product_labels:
        if label == selected_product:
            selected_product_code = code
            break
    
    # For Bolag, directly use the actual values from data, not through a mapping
    bolag_options = sorted(customer_data['Bolag'].unique().tolist())
    excluded_bolag_display = st.multiselect('Exclude Bolag', bolag_options)
    included_bolag = [b for b in bolag_options if b not in excluded_bolag_display]
    
    min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
    max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
    
    # Subject line input with GenAI checkbox
    col1, col2 = st.columns([3, 1])
    with col1:
        subject_line = st.text_input('Subject Line')
    with col2:
        use_genai = st.checkbox('GenAI', value=False)

    # Prediction logic
    if subject_line:
        # Create base input data
        base_input_data = pd.DataFrame(columns=features.columns)
        base_input_data.loc[0] = 0

        # Data validation - use the correct column names from dummy_df
        dialog_col = dummy_dialog_map.get(selected_dialog_code, f'Dialog_{selected_dialog_code}')
        syfte_col = dummy_syfte_map.get(selected_syfte_code, f'Syfte_{selected_syfte_code}')
        product_col = dummy_product_map.get(selected_product_code, f'Product_{selected_product_code}')
        
        # Check if columns exist
        if dialog_col not in features.columns:
            st.error(f"Selected Dialog '{selected_dialog_code}' maps to column '{dialog_col}' which is not found in features.")
            st.write(f"Available columns that start with 'Dialog_': {[c for c in features.columns if c.startswith('Dialog_')]}")
        elif syfte_col not in features.columns:
            st.error(f"Selected Syfte '{selected_syfte_code}' maps to column '{syfte_col}' which is not found in features.")
            st.write(f"Available columns that start with 'Syfte_': {[c for c in features.columns if c.startswith('Syfte_')]}")
        elif product_col not in features.columns:
            st.error(f"Selected Product '{selected_product_code}' maps to column '{product_col}' which is not found in features.")
            st.write(f"Available columns that start with 'Product_': {[c for c in features.columns if c.startswith('Product_')]}")
        else:
            base_input_data[dialog_col] = 1
            base_input_data[syfte_col] = 1
            base_input_data[product_col] = 1
            base_input_data['Min_age'] = min_age
            base_input_data['Max_age'] = max_age
            for b in included_bolag:
                bolag_col = f'Bolag_{b}'
                if bolag_col in base_input_data.columns:
                    base_input_data[bolag_col] = 1

            # Function to predict open rate
            def predict_for_subject(subject_line):
                input_data = base_input_data.copy()
                input_data['Subject_length'] = len(subject_line)
                input_data['Num_words'] = len(subject_line.split())
                input_data['Has_exclamation'] = 1 if '!' in subject_line else 0
                input_data['Has_question'] = 1 if '?' in subject_line else 0
                return model.predict(input_data)[0]

            # Calculate KPIs for current subject line
            openrate_A = predict_for_subject(subject_line)
            avg_clickrate = delivery_data['Clickrate'].mean()
            avg_optoutrate = delivery_data['Optoutrate'].mean()

            # Display predicted results for current subject line
            st.subheader('Predicted Results')
            col1, col2, col3 = st.columns(3)
            col1.metric("Open Rate", f"{openrate_A:.2%}")
            col2.metric("Click Rate", f"{avg_clickrate:.2%}")
            col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")

            # A/B/C/D Testing
            if use_genai and GROQ_API_KEY:
                # Create data to send to Groq
                groq_data = {
                    "subject": subject_line,
                    "metrics": {
                        "openrate": float(openrate_A),
                        "clickrate": avg_clickrate,
                        "optoutrate": avg_optoutrate
                    },
                    "targeting": {
                        "dialog": selected_dialog,
                        "syfte": selected_syfte,
                        "product": selected_product,
                        "age_range": {"min": min_age, "max": max_age}
                    },
                    "bolag": included_bolag
                }
                
                # Button to send to Groq API
                if st.button('Send to Groq API'):
                    st.session_state.groq_sending = True
                    st.info("Sending data to Groq API...")
                    
                    # Call Groq API
                    prompt = f"""
                    I need to create email subject lines for a marketing campaign. 
                    
                    Current subject line: "{subject_line}"
                    Predicted open rate: {openrate_A:.2%}
                    
                    Campaign details:
                    - Dialog: {selected_dialog}
                    - Syfte (Purpose): {selected_syfte}
                    - Product: {selected_product}
                    - Age range: {min_age} to {max_age}
                    - Target regions: {', '.join(included_bolag)}
                    
                    Please generate THREE alternative email subject lines in Swedish that could improve the open rate.
                    Return your response as a JSON object with a 'subject_lines' field containing an array of objects, each with a 'subject' field, like this:
                    {{
                        "subject_lines": [
                            {{"subject": "First alternative subject line"}},
                            {{"subject": "Second alternative subject line"}},
                            {{"subject": "Third alternative subject line"}}
                        ]
                    }}
                    """
                    
                    try:
                        # Make the API call here to get the response
                        response = requests.post(
                            'https://api.groq.com/openai/v1/chat/completions',
                            headers={
                                'Authorization': f'Bearer {GROQ_API_KEY}',
                                'Content-Type': 'application/json'
                            },
                            json={
                                "model": "llama3-70b-8192",
                                "messages": [
                                    {"role": "system", "content": "You are an expert in email marketing optimization. Your task is to generate compelling email subject lines in Swedish that maximize open rates."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0.7,
                                "response_format": {"type": "json_object"}
                            },
                            verify=False
                        )
                        
                        # Now process the response
                        if response.status_code == 200:
                            try:
                                response_data = response.json()
                                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                                print("Raw response content:", content)  # Debug print
                                
                                suggestions_data = json.loads(content)
                                print("Parsed suggestions data:", suggestions_data)  # Debug print
                                
                                # Extract suggestions correctly based on the response structure
                                if 'subject_lines' in suggestions_data:
                                    suggestions = suggestions_data['subject_lines']
                                elif isinstance(suggestions_data, list):
                                    suggestions = suggestions_data
                                else:
                                    # Fallback options if structure is unexpected
                                    st.warning("Unexpected response format. Falling back to simple alternatives.")
                                    suggestions = [
                                        {'subject': f"{subject_line}!"},
                                        {'subject': subject_line.title()},
                                        {'subject': ' '.join(subject_line.split()[:5])}
                                    ]
                                
                                print("Processed suggestions:", suggestions)  # Debug print
                                
                                # Process the suggestions for A/B/C/D testing
                                options = []
                                for i, sug in enumerate(suggestions[:3], start=1):
                                    subject = sug.get('subject', '')
                                    if subject:
                                        openrate = predict_for_subject(subject)
                                        options.append((chr(65 + i), subject, openrate))
                                
                                # Display alternative options with KPIs
                                st.subheader("Alternative Subject Lines")
                                
                                for opt, subject, openrate in options:
                                    st.write(f'**{opt}: "{subject}"** - Predicted Results:')
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Open Rate", f"{openrate:.2%}", f"{openrate - openrate_A:.2%}")
                                    col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                                    col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                                
                                # Determine best option
                                all_options = [('Current', subject_line, openrate_A)] + options
                                best_option = max(all_options, key=lambda x: x[2])
                                
                                st.subheader("Best Option")
                                if best_option[0] == 'Current':
                                    st.write(f'**Current subject line: "{best_option[1]}"**')
                                else:
                                    st.write(f'**{best_option[0]}: "{best_option[1]}"**')
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Open Rate", f"{best_option[2]:.2%}", f"{best_option[2] - openrate_A:.2%}" if best_option[0] != 'Current' else None)
                                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                                
                            except json.JSONDecodeError as e:
                                st.error(f'Failed to parse JSON response: {str(e)}')
                                st.code(response.text)  # Show the raw response for debugging
                        else:
                            st.error(f'API error: {response.status_code}')
                            st.code(response.text)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        # Show stack trace for detailed debugging
                        import traceback
                        st.code(traceback.format_exc())
                        
            elif use_genai and not GROQ_API_KEY:
                st.warning('Groq API key not found. Please set GROQ_API_KEY in .env file.')
            else:
                # Manual A/B/C/D testing without GenAI
                st.subheader('Alternative Subject Lines')
                
                # Generate simple alternatives without mentioning the original as option A
                subject_B = subject_line + '!'
                openrate_B = predict_for_subject(subject_B)
                st.write(f'**B: "{subject_B}"** - Predicted Results:')
                col1, col2, col3 = st.columns(3)
                col1.metric("Open Rate", f"{openrate_B:.2%}", f"{openrate_B - openrate_A:.2%}")
                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")

                subject_C = subject_line.title()
                openrate_C = predict_for_subject(subject_C)
                st.write(f'**C: "{subject_C}"** - Predicted Results:')
                col1, col2, col3 = st.columns(3)
                col1.metric("Open Rate", f"{openrate_C:.2%}", f"{openrate_C - openrate_A:.2%}")
                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")

                subject_D = ' '.join(subject_line.split()[:5])
                openrate_D = predict_for_subject(subject_D)
                st.write(f'**D: "{subject_D}"** - Predicted Results:')
                col1, col2, col3 = st.columns(3)
                col1.metric("Open Rate", f"{openrate_D:.2%}", f"{openrate_D - openrate_A:.2%}")
                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")

                all_options = [
                    ('Current', subject_line, openrate_A),
                    ('B', subject_B, openrate_B),
                    ('C', subject_C, openrate_C),
                    ('D', subject_D, openrate_D)
                ]
                best_option = max(all_options, key=lambda x: x[2])
                
                st.subheader("Best Option")
                if best_option[0] == 'Current':
                    st.write(f'**Current subject line: "{best_option[1]}"**')
                else:
                    st.write(f'**{best_option[0]}: "{best_option[1]}"**')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Open Rate", f"{best_option[2]:.2%}", f"{best_option[2] - openrate_A:.2%}" if best_option[0] != 'Current' else None)
                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
