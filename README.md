# PredictKPI
# Multi-KPI Email Campaign Optimizer

A comprehensive application for optimizing email campaigns by predicting and analyzing multiple Key Performance Indicators (KPIs) such as open rates, click rates, and opt-out rates.

## Features

- **Multi-KPI Prediction**: Automatically selects the best model for each KPI metric (open rate, click rate, opt-out rate)
- **Subject Line & Preheader Optimization**: Evaluates multiple content options to maximize engagement
- **AI-Powered Suggestions**: Integrates with Groq API to generate optimized subject line alternatives
- **Batch Processing**: Analyze multiple subject lines at once for campaign planning
- **Interactive Dashboards**: Visualize KPI trends and performance by campaign categories
- **Model Comparison**: Track model improvements and feature importance

## Architecture

This application implements a hierarchical modeling approach:

1. **Feature Engineering Layer**: Creates specialized features for each KPI type
2. **Model Selection Layer**: Automatically evaluates multiple model types (XGBoost, LightGBM, CatBoost) for each KPI
3. **Content Optimization Layer**: Evaluates content alternatives using the best models
4. **Visualization Layer**: Presents results through interactive dashboards

## Data Preparation

Place your data files in the `Data` directory:

- `delivery_data.csv`: Email campaign data with the following columns:
  - `InternalName`: Delivery identifier
  - `Subject`: Email subject line
  - `Preheader`: Email preheader
  - `Date`: Date and time of delivery
  - `Sendouts`: Total sendout count
  - `Opens`: Total opens count
  - `Clicks`: Total clicks count
  - `Optouts`: Total unsubscribe count
  - `Dialog`, `Syfte`, `Product`: Campaign metadata

- `customer_data.csv`: Customer data with the following columns:
  - `Primary key`: Customer identifier
  - `InternalName`: Delivery identifier (to link with delivery data)
  - `OptOut`: Opt-out in delivery (1/0)
  - `Open`: Opened delivery (1/0)
  - `Click`: Clicked in delivery (1/0)
  - `Gender`: Customer gender
  - `Age`: Customer age
  - `Bolag`: Customer company connection

## Model Training

The system automatically selects the best model for each KPI:

1. On first run, the system will train models for each KPI
2. Each model version is saved for future use
3. You can force retraining by clicking "Force model retraining" in the sidebar
4. Advanced model settings can be configured in the sidebar

## Acknowledgments

- [Streamlit](https://streamlit.io/) - The web framework used
- [XGBoost](https://xgboost.readthedocs.io/) - For gradient boosting models
- [LightGBM](https://lightgbm.readthedocs.io/) - For gradient boosting models
- [CatBoost](https://catboost.ai/) - For gradient boosting models
- [Groq](https://groq.com/) - For AI-powered subject line suggestions