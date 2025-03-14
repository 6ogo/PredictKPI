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

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Data Loading and Preprocessing ---
delivery_data = pd.read_csv('delivery_data.csv', sep=';', encoding='utf-8')
customer_data = pd.read_csv('customer_data.csv', sep=';', encoding='utf-8')

# Calculate KPIs in delivery_data
delivery_data['Openrate'] = delivery_data['Opens'] / delivery_data['Sendouts']
delivery_data['Clickrate'] = delivery_data['Clicks'] / delivery_data['Sendouts']
delivery_data['Optoutrate'] = delivery_data['Optouts'] / delivery_data['Sendouts']

# Aggregate customer data to get age span per InternalName
age_stats = customer_data.groupby('InternalName')['Age'].agg(['min', 'max']).reset_index()
age_stats.columns = ['InternalName', 'Min_age', 'Max_age']
delivery_data = delivery_data.merge(age_stats, on='InternalName', how='left')

# Aggregate Bolag (company) data as binary features per InternalName
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

# Prepare features and target for XGBoost
features = pd.concat([
    pd.get_dummies(delivery_data[categorical_features]),
    delivery_data[numerical_features],
    delivery_data[bolag_features_list]
], axis=1)
target = delivery_data['Openrate']

# Train XGBoost model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# --- Streamlit App ---
st.title('Sendout KPI Predictor')

# Create tabs
tab1, tab2 = st.tabs(['Model Results', 'Sendout Prediction'])

# Tab 1: Model Results
with tab1:
    st.header('Model Performance')
    st.write(f'Mean Squared Error on test set: {mse:.4f}')
    # Feature importances plot
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

    # Get unique options for dropdowns
    dialog_options = delivery_data['Dialog'].unique().tolist()
    syfte_options = delivery_data['Syfte'].unique().tolist()
    produkt_options = delivery_data['Produkt'].unique().tolist()
    bolag_options = customer_data['Bolag'].unique().tolist()

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
        # Create base input data (without subject line features)
        base_input_data = pd.DataFrame(columns=features.columns)
        base_input_data.loc[0] = 0
        base_input_data['Dialog_' + selected_dialog] = 1
        base_input_data['Syfte_' + selected_syfte] = 1
        base_input_data['Produkt_' + selected_produkt] = 1
        base_input_data['Min_age'] = min_age
        base_input_data['Max_age'] = max_age
        for b in included_bolag:
            if 'Bolag_' + b in base_input_data.columns:
                base_input_data['Bolag_' + b] = 1

        # Function to predict open rate for a subject line
        def predict_for_subject(subject_line):
            input_data = base_input_data.copy()
            input_data['Subject_length'] = len(subject_line)
            input_data['Num_words'] = len(subject_line.split())
            input_data['Has_exclamation'] = 1 if '!' in subject_line else 0
            input_data['Has_question'] = 1 if '?' in subject_line else 0
            return model.predict(input_data)[0]

        # A/B/C/D Testing
        st.subheader('Subject Line A/B/C/D Testing')

        # Option A: User-input subject line
        openrate_A = predict_for_subject(subject_line)
        st.write(f'**A: "{subject_line}"** - Predicted Openrate: {openrate_A:.2%}')

        if use_genai and GROQ_API_KEY:
            if st.button('Send to GenAI'):
                # Prepare prompt for Groq API
                prompt = (
                    f"Generate three alternative email subject lines based on the following user input: '{subject_line}'. "
                    "The subject lines should be designed to maximize open rates for a marketing email campaign. "
                    "For each suggestion, provide only the subject line text. "
                    "Return the response in JSON format with a list of dictionaries, each containing the field 'subject'."
                )

                # API call to Groq
                response = requests.post(
                    'https://api.groq.com/generate',  # Hypothetical endpoint
                    headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
                    json={'prompt': prompt}
                )

                if response.status_code == 200:
                    suggestions = response.json()
                    options = []
                    for i, sug in enumerate(suggestions, start=1):
                        subject = sug['subject']
                        openrate = predict_for_subject(subject)
                        options.append((chr(65 + i), subject, openrate))

                    # Display GenAI suggestions
                    for opt, subject, openrate in options:
                        st.write(f'**{opt}: "{subject}"** - Predicted Openrate: {openrate:.2%}')

                    # Find the best option
                    all_options = [('A', subject_line, openrate_A)] + options
                    best_option = max(all_options, key=lambda x: x[2])
                    st.write(f'**Best Option: {best_option[0]} - "{best_option[1]}"** with Predicted Openrate: {best_option[2]:.2%}')
                else:
                    st.error('Failed to fetch suggestions from Groq API.')
        elif use_genai and not GROQ_API_KEY:
            st.warning('Groq API key not found. Please set the GROQ_API_KEY in the .env file.')

        # Simulated variations (fallback if not using GenAI or API key is missing)
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

            # Find the best simulated option
            all_options = [
                ('A', subject_line, openrate_A),
                ('B', subject_B, openrate_B),
                ('C', subject_C, openrate_C),
                ('D', subject_D, openrate_D)
            ]
            best_option = max(all_options, key=lambda x: x[2])
            st.write(f'**Best Option: {best_option[0]} - "{best_option[1]}"** with Predicted Openrate: {best_option[2]:.2%}')
