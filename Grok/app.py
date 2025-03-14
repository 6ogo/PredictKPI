import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
categorical_features = ['Dialog', 'Syfte', 'Produkt']
numerical_features = ['Min_age', 'Max_age', 'Subject_length', 'Num_words', 'Has_exclamation', 'Has_question']
bolag_features_list = [col for col in delivery_data.columns if col.startswith('Bolag_')]

# Prepare features and target
features = pd.concat([
    pd.get_dummies(delivery_data[categorical_features]),
    delivery_data[numerical_features],
    delivery_data[bolag_features_list]
], axis=1)
target = delivery_data['Openrate']

# Load or train XGBoost model
model_file = 'xgboost_model.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

# --- Define Enums ---
BOLAG_VALUES = {
    "Blekinge": "B02", "Dalarna": "B03", "Älvsborg": "B04", "Gävleborg": "B08",
    "Göinge-Kristianstad": "B09", "Göteborg-Bohuslan": "B10", "Halland": "B11",
    "Jämtland": "B14", "Jönköping": "B15", "Kalmar": "B16", "Kronoberg": "B21",
    "Norrbotten": "B24", "Skaraborg": "B27", "Stockholm": "B28", "Södermanland": "B29",
    "Uppsala": "B31", "Värmland": "B32", "Västerbotten": "B34", "Västernorrland": "B35",
    "Bergslagen": "B37", "Östgöta": "B42", "Gotland": "B43", "Skåne": "B50"
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
available_produkt = delivery_data['Produkt'].unique().tolist()

bolag_options = [key for key, value in BOLAG_VALUES.items() if value in available_bolag]
syfte_options = [SYFTE_VALUES[key][1] for key in SYFTE_VALUES if SYFTE_VALUES[key][0] in available_syfte]
dialog_options = [DIALOG_VALUES[key][1] for key in DIALOG_VALUES if DIALOG_VALUES[key][0] in available_dialog]
produkt_options = [PRODUKT_VALUES[key][1] for key in PRODUKT_VALUES if PRODUKT_VALUES[key][0] in available_produkt]

# --- Streamlit App ---
st.title('Sendout KPI Predictor')

# Create tabs
tab1, tab2 = st.tabs(['Model Results', 'Sendout Prediction'])

# Tab 1: Model Results
with tab1:
    st.header('Model Performance')
    st.write(f'Mean Squared Error on test set: {mse:.4f}')
    importances = model.feature_importances_
    feature_names = features.columns
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_names, importances)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    plt.tight_layout()
    st.pyplot(fig)

# Tab 2: Sendout Prediction
with tab2:
    st.header('Predict KPIs for New Sendout')

    # User input fields
    selected_dialog = st.selectbox('Dialog', dialog_options)
    selected_syfte = st.selectbox('Syfte', syfte_options)
    selected_produkt = st.selectbox('Produkt', produkt_options)
    min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
    max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
    excluded_bolag = st.multiselect('Exclude Bolag', bolag_options)
    included_bolag = [b for b in bolag_options if b not in excluded_bolag]

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

        # Data validation
        dialog_col = 'Dialog_' + selected_dialog
        syfte_col = 'Syfte_' + selected_syfte
        produkt_col = 'Produkt_' + selected_produkt

        if dialog_col not in features.columns:
            st.error(f"Selected Dialog '{selected_dialog}' not found in training data.")
        elif syfte_col not in features.columns:
            st.error(f"Selected Syfte '{selected_syfte}' not found in training data.")
        elif produkt_col not in features.columns:
            st.error(f"Selected Produkt '{selected_produkt}' not found in training data.")
        else:
            base_input_data[dialog_col] = 1
            base_input_data[syfte_col] = 1
            base_input_data[produkt_col] = 1
            base_input_data['Min_age'] = min_age
            base_input_data['Max_age'] = max_age
            for b in included_bolag:
                if 'Bolag_' + BOLAG_VALUES[b] in base_input_data.columns:
                    base_input_data['Bolag_' + BOLAG_VALUES[b]] = 1

            # Function to predict open rate
            def predict_for_subject(subject_line):
                input_data = base_input_data.copy()
                input_data['Subject_length'] = len(subject_line)
                input_data['Num_words'] = len(subject_line.split())
                input_data['Has_exclamation'] = 1 if '!' in subject_line else 0
                input_data['Has_question'] = 1 if '?' in subject_line else 0
                return model.predict(input_data)[0]

            # A/B/C/D Testing
            st.subheader('Subject Line A/B/C/D Testing')
            openrate_A = predict_for_subject(subject_line)
            st.write(f'**A: "{subject_line}"** - Predicted Openrate: {openrate_A:.2%}')

            if use_genai and GROQ_API_KEY:
                if st.button('Send to GenAI'):
                    prompt = (
                        f"Generate three alternative email subject lines in Swedish based on the following user input: '{subject_line}'. "
                        "The subject lines should be designed to maximize open rates for a marketing email campaign. "
                        "Return the response in JSON format as a list of dictionaries, each containing the field 'subject', exactly like this: "
                        "[{'subject': 'Summer Deals Await You!'}, {'subject': 'Don’t Miss Our Summer Sale!'}, {'subject': 'Hot Summer Savings Start Now'}]."
                    )

                    response = requests.post(
                        'https://api.groq.com/openai/v1/chat/completions',  # Hypothetical endpoint
                        headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
                        json={'prompt': prompt}
                    )

                    if response.status_code == 200:
                        try:
                            suggestions = response.json()
                        except json.JSONDecodeError:
                            st.error('Failed to parse JSON response from Groq API.')
                            suggestions = []
                        options = []
                        for i, sug in enumerate(suggestions[:3], start=1):
                            subject = sug.get('subject', '')
                            openrate = predict_for_subject(subject)
                            options.append((chr(65 + i), subject, openrate))
                        for opt, subject, openrate in options:
                            st.write(f'**{opt}: "{subject}"** - Predicted Openrate: {openrate:.2%}')
                        all_options = [('A', subject_line, openrate_A)] + options
                        best_option = max(all_options, key=lambda x: x[2])
                        st.write(f'**Best Option: {best_option[0]} - "{best_option[1]}"** with Predicted Openrate: {best_option[2]:.2%}')
                    else:
                        st.error('Failed to fetch suggestions from Groq API.')
            elif use_genai and not GROQ_API_KEY:
                st.warning('Groq API key not found. Please set GROQ_API_KEY in .env file.')
            else:
                subject_B = subject_line + '!'
                openrate_B = predict_for_subject(subject_B)
                st.write(f'**B: "{subject_B}"** - Predicted Openrate: {openrate_B:.2%}')

                subject_C = subject_line.title()
                openrate_C = predict_for_subject(subject_C)
                st.write(f'**C: "{subject_C}"** - Predicted Openrate: {openrate_C:.2%}')

                subject_D = ' '.join(subject_line.split()[:5])
                openrate_D = predict_for_subject(subject_D)
                st.write(f'**D: "{subject_D}"** - Predicted Openrate: {openrate_D:.2%}')

                all_options = [
                    ('A', subject_line, openrate_A),
                    ('B', subject_B, openrate_B),
                    ('C', subject_C, openrate_C),
                    ('D', subject_D, openrate_D)
                ]
                best_option = max(all_options, key=lambda x: x[2])
                st.write(f'**Best Option: {best_option[0]} - "{best_option[1]}"** with Predicted Openrate: {best_option[2]:.2%}')