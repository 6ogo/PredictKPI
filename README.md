# PredictKPI
A Python Streamlit application that predicts Key Performance Indicators (KPIs) from past email marketing campaign results. This tool helps marketers optimize their email campaigns by predicting open rates, click rates, and opt-out rates based on historical data.

## Features
- **KPI Prediction**: Predict open rates, click rates, and opt-out rates for new email campaigns
- **AI-Powered Subject Line Optimization**: Generate and test alternative subject lines using Groq LLM API
- **A/B/C/D Testing**: Compare your subject line against AI-generated alternatives
- **Age Group Analysis**: Visualize how different age groups respond to your campaigns
- **Model Training**: Train, evaluate, and manage different model versions
- **Feature Importance Analysis**: Understand what factors influence your email performance
- **Interactive Visualizations**: Explore data through heatmaps and charts

## Data Requirements
The application expects two main data files:

### Delivery Data (delivery_data.csv)
CSV file with semicolon (`;`) separator containing:

- `InternalName`: Delivery identifier
- `Subject`: Email subject line
- `Date`: Date and time of delivery
- `Sendouts`: Total number of emails sent
- `Opens`: Total number of opens
- `Clicks`: Total number of clicks
- `Optouts`: Total number of unsubscribes
- `Dialog`, `Syfte`, `Product`: Campaign metadata
- `Preheader`: Email preheader (for v2.0.0+ models)

Example:
```
InternalName;Subject;Date;Sendouts;Opens;Clicks;Optouts;Dialog;Syfte;Product
DM123456;Take the car to your next adventure;2024/06/10 15:59;14827;2559;211;9;F;VD;Mo
```

### Customer Data (customer_data.csv)
CSV file with semicolon (`;`) separator containing:

- `Primary key`: Customer identifier
- `InternalName`: Delivery identifier to link with delivery data
- `OptOut`: If customer opted out (1/0)
- `Open`: If customer opened the email (1/0)
- `Click`: If customer clicked in the email (1/0)
- `Gender`: Customer gender
- `Age`: Customer age
- `Bolag`: Customer company/region connection

Example:
```
Primary key;OptOut;Open;Click;Gender;Age;InternalName;Bolag
12345678;0;1;0;Kvinna;69;DM123456;Stockholm
```

## Model Versions
The application uses semantic versioning (Major.Minor.Patch) for models:

- **v1.x.x**: Basic models with subject line features only
- **v2.x.x**: Enhanced models with both subject line and preheader features

Each model version has its own documentation and performance metrics saved in the `Docs/model_vX.X.X/` directory.

## Key Components
- **Feature Engineering**: Extracts features from subject lines, preheaders, and campaign metadata
- **XGBoost Model**: Machine learning model to predict email performance
- **Groq API Integration**: Generates optimized subject line alternatives
- **Age Group Analysis**: Segments performance by customer age groups
- **Model Versioning**: Manages multiple model versions with performance documentation

## Configuration
The application supports various configuration options:

- **Data Sources**: Adjust file paths in the `load_data()` function
- **Model Parameters**: Configure hyperparameters when training new models
- **Sample Weights**: Adjust how the model weights high-performing campaigns
- **Age Grouping**: Modify age group definitions in the `categorize_age()` function

## Project Structure
```
PredictKPI/
├── Data/
│   ├── customer_data.csv      # Customer-level data
│   ├── delivery_data.csv      # Delivery-level data
│   └── example_*.csv          # Example data files
├── app/
│   ├── app.py                 # Main Streamlit application
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Saved model files
│   └── Docs/                  # Model documentation
├── Documentation.md           # Detailed documentation
├── LICENSE                    # MIT License
├── README.md                  # This file
└── example.env                # Example environment variables
```

## Advanced Features
### Model Training

Train new model versions with customized parameters:
1. Navigate to the "Model Results" tab
2. Expand "Retrain Model with Custom Parameters"
3. Adjust model parameters and sample weight configuration
4. Click "Retrain Model"

### Age Group Analysis
Analyze how different age groups interact with your campaigns:
1. Navigate to the "Model Results" tab
2. Expand "Age Group Analysis"
3. Select which views to display (Overall, Dialog, Syfte, Product)
4. Compare open rates, click rates, and opt-out rates across age groups

### A/B/C/D Testing with AI
Test your subject line against AI-generated alternatives:
1. Navigate to the "Sendout Prediction" tab
2. Enter your subject line (and preheader for v2+ models)
3. Check the "GenAI" box
4. Click "Send to Groq API"
5. Compare the predicted performance of all versions

## Acknowledgements
- [Streamlit](https://streamlit.io/) for the interactive web application framework
- [XGBoost](https://xgboost.readthedocs.io/) for the gradient boosting framework
- [Groq](https://groq.com/) for the LLM API powering subject line generation
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/) for visualizations
