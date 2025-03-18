import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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

# **Extract subject line and preheader features**
delivery_data['Subject_length'] = delivery_data['Subject'].str.len()
delivery_data['Subject_num_words'] = delivery_data['Subject'].str.split().str.len()
delivery_data['Subject_has_exclamation'] = delivery_data['Subject'].str.contains('!').astype(int)
delivery_data['Subject_has_question'] = delivery_data['Subject'].str.contains(r'\?', regex=True).astype(int)

delivery_data['Preheader_length'] = delivery_data['Preheader'].str.len()
delivery_data['Preheader_num_words'] = delivery_data['Preheader'].str.split().str.len()
delivery_data['Preheader_has_exclamation'] = delivery_data['Preheader'].str.contains('!').astype(int)
delivery_data['Preheader_has_question'] = delivery_data['Preheader'].str.contains(r'\?', regex=True).astype(int)

# Define feature columns
categorical_features = ['Dialog', 'Syfte', 'Product']
numerical_features = [
    'Min_age', 'Max_age', 
    'Subject_length', 'Subject_num_words', 'Subject_has_exclamation', 'Subject_has_question',
    'Preheader_length', 'Preheader_num_words', 'Preheader_has_exclamation', 'Preheader_has_question'
]
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
dummy_dialog_map = {dialog: f'Dialog_{dialog}' for dialog in delivery_data['Dialog'].unique()}
dummy_syfte_map = {syfte: f'Syfte_{syfte}' for syfte in delivery_data['Syfte'].unique()}
dummy_product_map = {product: f'Product_{product}' for product in delivery_data['Product'].unique()}

print("Dialog mapping:", dummy_dialog_map)
print("Syfte mapping:", dummy_syfte_map)
print("Product mapping:", dummy_product_map)

# **Split data into training and test sets (moved outside conditional)**
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# **Define sample weights**
THRESHOLD = 0.5
WEIGHT_HIGH = 2.0
WEIGHT_LOW = 1.0
sample_weights_train = np.where(y_train > THRESHOLD, WEIGHT_HIGH, WEIGHT_LOW)

# Load or train XGBoost model with regularization and sample weights
model_file = 'xgboost_model.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
    mse = 0.0  # Default value since we're loading a pre-trained model
    st.write("Loaded existing model from", model_file)
else:
    # Initialize model with L2 regularization
    model = XGBRegressor(reg_lambda=1.0, random_state=42)
    # Train with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights_train)
    joblib.dump(model, model_file)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write("Trained and saved new model to", model_file)

# **Cross-Validation on Training Set**
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse_scores = []
cv_rmse_scores = []
cv_mae_scores = []
cv_r2_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    sample_weights_fold = np.where(y_train_fold > THRESHOLD, WEIGHT_HIGH, WEIGHT_LOW)
    model_cv = XGBRegressor(reg_lambda=1.0, random_state=42)
    model_cv.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
    y_pred_val = model_cv.predict(X_val_fold)
    mse_cv = mean_squared_error(y_val_fold, y_pred_val)
    rmse_cv = np.sqrt(mse_cv)
    mae_cv = mean_absolute_error(y_val_fold, y_pred_val)
    r2_cv = r2_score(y_val_fold, y_pred_val)
    cv_mse_scores.append(mse_cv)
    cv_rmse_scores.append(rmse_cv)
    cv_mae_scores.append(mae_cv)
    cv_r2_scores.append(r2_cv)

avg_cv_mse = np.mean(cv_mse_scores)
avg_cv_rmse = np.mean(cv_rmse_scores)
avg_cv_mae = np.mean(cv_mae_scores)
avg_cv_r2 = np.mean(cv_r2_scores)

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

print("Available Bolag:", available_bolag)
print("Available Syfte:", available_syfte)
print("Available Dialog:", available_dialog)
print("Available Product:", available_produkt)

# Check bolag mapping
bolag_options = [key for key, value in BOLAG_VALUES.items() if value in available_bolag]

# --- Streamlit App ---
st.title('Sendout KPI Predictor')

# Create tabs
tab1, tab2 = st.tabs(['Sendout Prediction', 'Model Results'])

# Tab 2: Model Performance
with tab2:
    st.header('Model Performance')
    
    if os.path.exists(model_file):
        st.info("Model was loaded from saved file. Let's verify its performance on the dataset.")
        
        X_verify = features
        y_verify = target
        y_pred = model.predict(X_verify)
        
        mse = mean_squared_error(y_verify, y_pred)
        mae = np.mean(np.abs(y_verify - y_pred))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_verify, y_pred)
        
        # **Add Cross-Validation and Test Set Results**
        st.subheader("Cross-Validation Performance (5-fold on Training Set)")
        col1, col2 = st.columns(2)
        col1.metric("Average Mean Squared Error", f"{avg_cv_mse:.6f}")
        col1.metric("Average Root MSE", f"{avg_cv_rmse:.6f}")
        col2.metric("Average Mean Absolute Error", f"{avg_cv_mae:.6f}")
        col2.metric("Average R² Score", f"{avg_cv_r2:.4f}")
        
        st.subheader("Test Set Performance")
        y_pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse_test:.6f}")
        col1.metric("Root MSE", f"{rmse_test:.6f}")
        col2.metric("Mean Absolute Error", f"{mae_test:.6f}")
        col2.metric("R² Score", f"{r2_test:.4f}")
        
        st.subheader("Full Dataset Verification")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.6f}")
        col1.metric("Root MSE", f"{rmse:.6f}")
        col2.metric("Mean Absolute Error", f"{mae:.6f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        feature_names = features.columns
        
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
        
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {X_verify.shape[0]}")
        st.write(f"Number of features: {X_verify.shape[1]}")
        
        st.subheader("Predictions vs Actual Values (Test Set)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel('Actual Open Rate')
        ax.set_ylabel('Predicted Open Rate')
        ax.set_title('Actual vs Predicted Open Rates (Test Set)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Distribution of Prediction Errors (Test Set)")
        errors = y_test - y_pred_test
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors (Test Set)')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        st.write(f'Training set size: {X_train.shape[0]} samples')
        st.write(f'Test set size: {X_test.shape[0]} samples')
        st.write(f'Mean Squared Error on test set: {mse:.6f}')
        st.write(f'Root Mean Squared Error: {np.sqrt(mse):.6f}')
        st.write(f'Mean Absolute Error: {np.mean(np.abs(y_test - model.predict(X_test))):.6f}')
        st.write(f'R² Score: {r2_score(y_test, model.predict(X_test)):.4f}')
        
        # **Add Cross-Validation Results**
        st.subheader("Cross-Validation Performance (5-fold on Training Set)")
        col1, col2 = st.columns(2)
        col1.metric("Average Mean Squared Error", f"{avg_cv_mse:.6f}")
        col1.metric("Average Root MSE", f"{avg_cv_rmse:.6f}")
        col2.metric("Average Mean Absolute Error", f"{avg_cv_mae:.6f}")
        col2.metric("Average R² Score", f"{avg_cv_r2:.4f}")
        
        importances = model.feature_importances_
        feature_names = features.columns
        
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
        
        st.dataframe(importance_df)

# Tab 1: Sendout Prediction
with tab1:
    st.header('Predict KPIs for New Sendout')

    dialog_options = sorted(delivery_data['Dialog'].unique().tolist())
    dialog_labels = [(d, next((value[1] for key, value in DIALOG_VALUES.items() if value[0] == d), d)) for d in dialog_options]
    selected_dialog = st.selectbox('Dialog', options=[label for _, label in dialog_labels], format_func=lambda x: x)
    selected_dialog_code = next(code for code, label in dialog_labels if label == selected_dialog)
    
    syfte_options = sorted(delivery_data['Syfte'].unique().tolist())
    syfte_labels = [(s, next((value[1] for key, value in SYFTE_VALUES.items() if value[0] == s), s)) for s in syfte_options]
    selected_syfte = st.selectbox('Syfte', options=[label for _, label in syfte_labels], format_func=lambda x: x)
    selected_syfte_code = next(code for code, label in syfte_labels if label == selected_syfte)
    
    product_options = sorted(delivery_data['Product'].unique().tolist())
    product_labels = [(p, next((value[1] for key, value in PRODUKT_VALUES.items() if value[0] == p), p)) for p in product_options]
    selected_product = st.selectbox('Product', options=[label for _, label in product_labels], format_func=lambda x: x)
    selected_product_code = next(code for code, label in product_labels if label == selected_product)
    
    bolag_options = sorted(customer_data['Bolag'].unique().tolist())
    excluded_bolag_display = st.multiselect('Exclude Bolag', bolag_options)
    included_bolag = [b for b in bolag_options if b not in excluded_bolag_display]
    
    min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
    max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
    
    # **Subject line and Preheader input with GenAI checkbox**
    col1, col2 = st.columns([3, 1])
    with col1:
        subject_line = st.text_input('Subject Line')
        preheader = st.text_input('Preheader')
    with col2:
        use_genai = st.checkbox('GenAI', value=False)

    # Prediction logic
    if subject_line and preheader:
        # Create base input data
        base_input_data = pd.DataFrame(columns=features.columns)
        base_input_data.loc[0] = 0

        dialog_col = dummy_dialog_map.get(selected_dialog_code, f'Dialog_{selected_dialog_code}')
        syfte_col = dummy_syfte_map.get(selected_syfte_code, f'Syfte_{selected_syfte_code}')
        product_col = dummy_product_map.get(selected_product_code, f'Product_{selected_product_code}')
        
        if dialog_col not in features.columns:
            st.error(f"Selected Dialog '{selected_dialog_code}' maps to column '{dialog_col}' which is not found in features.")
            st.write(f"Available columns: {[c for c in features.columns if c.startswith('Dialog_')]}")
        elif syfte_col not in features.columns:
            st.error(f"Selected Syfte '{selected_syfte_code}' maps to column '{syfte_col}' which is not found in features.")
            st.write(f"Available columns: {[c for c in features.columns if c.startswith('Syfte_')]}")
        elif product_col not in features.columns:
            st.error(f"Selected Product '{selected_product_code}' maps to column '{product_col}' which is not found in features.")
            st.write(f"Available columns: {[c for c in features.columns if c.startswith('Product_')]}")
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

            # **Updated prediction function for Subject and Preheader**
            def predict_for_subject_and_preheader(subject_line, preheader):
                input_data = base_input_data.copy()
                input_data['Subject_length'] = len(subject_line)
                input_data['Subject_num_words'] = len(subject_line.split())
                input_data['Subject_has_exclamation'] = 1 if '!' in subject_line else 0
                input_data['Subject_has_question'] = 1 if '?' in subject_line else 0
                input_data['Preheader_length'] = len(preheader)
                input_data['Preheader_num_words'] = len(preheader.split())
                input_data['Preheader_has_exclamation'] = 1 if '!' in preheader else 0
                input_data['Preheader_has_question'] = 1 if '?' in preheader else 0
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

            # **A/B/C/D Testing with Groq API including Preheader**
            if use_genai and GROQ_API_KEY:
                groq_data = {
                    "subject": subject_line,
                    "preheader": preheader,
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
                
                if st.button('Send to Groq API'):
                    st.session_state.groq_sending = True
                    st.info("Sending data to Groq API...")
                    
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
                    
                    try:
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
                            verify=False
                        )
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                            suggestions_data = json.loads(content)
                            
                            suggestions = suggestions_data.get('suggestions', [])
                            
                            options = []
                            for i, sug in enumerate(suggestions[:3], start=1):
                                subject = sug.get('subject', '')
                                preheader = sug.get('preheader', '')
                                if subject and preheader:
                                    openrate = predict_for_subject_and_preheader(subject, preheader)
                                    options.append((chr(65 + i), subject, preheader, openrate))
                            
                            st.subheader("Alternative Subject Lines and Preheaders")
                            for opt, subject, preheader, openrate in options:
                                st.write(f'**{opt}: Subject: "{subject}", Preheader: "{preheader}"** - Predicted Results:')
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Open Rate", f"{openrate:.2%}", f"{openrate - openrate_A:.2%}")
                                col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                                col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                            
                            all_options = [('Current', subject_line, preheader, openrate_A)] + options
                            best_option = max(all_options, key=lambda x: x[3])
                            
                            st.subheader("Best Option")
                            if best_option[0] == 'Current':
                                st.write(f'**Current: Subject: "{best_option[1]}", Preheader: "{best_option[2]}"**')
                            else:
                                st.write(f'**{best_option[0]}: Subject: "{best_option[1]}", Preheader: "{best_option[2]}"**')
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Open Rate", f"{best_option[3]:.2%}", f"{best_option[3] - openrate_A:.2%}" if best_option[0] != 'Current' else None)
                            col2.metric("Click Rate", f"{avg_clickrate:.2%}")
                            col3.metric("Opt-out Rate", f"{avg_optoutrate:.2%}")
                        else:
                            st.error(f'API error: {response.status_code}')
                            st.code(response.text)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            elif use_genai and not GROQ_API_KEY:
                st.warning('Groq API key not found. Please set GROQ_API_KEY in .env file.')