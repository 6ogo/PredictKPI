# PredictKPI
Python streamlit app to predict KPI from past email sendout results.

## How the code works
Loads the Groq API key from an .env file.
Adds a "GenAI" checkbox to the right of the subject line input.
Activates a "Send to GenAI" button when the checkbox is selected, sending an optional request to the Groq API.
Fetches three alternative subject line suggestions from the API.
Predicts the open rates for the original subject line and the three suggestions using an XGBoost model.
Displays a comparison of all four options (A, B, C, D) and identifies the best-performing subject line based on the highest predicted open rate.
