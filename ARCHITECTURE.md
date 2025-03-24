# Multi-KPI Email Campaign Optimizer: System Architecture

This document outlines the architecture and design decisions for the Multi-KPI Email Campaign Optimizer.

## System Overview

The system is designed to predict and optimize multiple KPIs (Key Performance Indicators) for email campaigns, focusing on open rates, click rates, and opt-out rates. It uses a hierarchical modeling approach where each KPI has its own dedicated model and feature engineering pipeline.

## Core Components

### 1. Data Management Layer

**Key Files:**
- `utils.py` - Data loading and preprocessing utilities

**Responsibilities:**
- Loading and validating input data
- Preprocessing raw data
- Maintaining constants and mappings
- Managing model versioning

### 2. Feature Engineering Layer

**Key Files:**
- `enhanced_features.py` - Specialized feature engineering for different KPIs

**Responsibilities:**
- Transforming raw data into model-ready features
- Creating specialized features for each KPI type:
  - **Open Rate Features**: Focus on subject line characteristics, send time, audience demographics
  - **Click Rate Features**: Include open rate as a feature, focus on call-to-action elements
  - **Opt-out Rate Features**: Focus on potentially annoying content patterns, previous engagement

### 3. Model Management Layer

**Key Files:**
- `multi_kpi_models.py` - Core model training and selection logic

**Responsibilities:**
- Training and evaluating multiple model types for each KPI
- Automatic model selection based on performance metrics
- Saving and loading models with versioning
- Making predictions for single campaigns or batches
- Evaluating subject line and preheader alternatives

### 4. User Interface Layer

**Key Files:**
- `app.py` - Main application entry point
- `ui_components.py` - Reusable UI components

**Responsibilities:**
- Providing intuitive user interfaces for various tasks
- Visualizing predictions and model performance
- Enabling batch processing of subject lines
- Displaying interactive dashboards for KPI analysis

### 5. Integration Layer

**Key Files:**
- Integration with Groq API for AI-powered suggestions

**Responsibilities:**
- Communicating with external APIs
- Processing API responses
- Error handling for external service failures

## Data Flow

1. **Input Data Loading**:
   - Load delivery data and customer data
   - Validate required columns
   - Calculate basic KPIs (open rate, click rate, opt-out rate)

2. **Feature Engineering**:
   - Transform raw data into features for each KPI type
   - Apply specialized feature engineering for each KPI

3. **Model Training/Selection**:
   - Train multiple model types for each KPI
   - Evaluate models using appropriate metrics
   - Select best model for each KPI

4. **Prediction & Optimization**:
   - Make predictions using the selected models
   - Generate and evaluate content alternatives
   - Select optimal content based on predicted KPIs

5. **Visualization & Presentation**:
   - Display predictions and insights
   - Visualize model performance
   - Show KPI trends and patterns

## Model Selection Strategy

The system implements a competitive model selection strategy:

1. For each KPI (open rate, click rate, opt-out rate):
   - Train multiple model types (XGBoost, LightGBM, CatBoost)
   - Evaluate each model using appropriate metrics (R², RMSE)
   - Select the best performing model for that KPI
   - Save the selected model with metadata

2. The best model is determined primarily by R² score, which measures how well the model explains the variance in the target variable.

## Content Optimization Strategy

The content optimization process follows these steps:

1. Predict KPIs for the baseline content (user-provided subject line and preheader)
2. Generate alternative content using the Groq API
3. Predict KPIs for all alternatives using the appropriate models
4. Calculate a combined score that balances:
   - Maximizing open rate
   - Maximizing click rate
   - Minimizing opt-out rate
5. Rank alternatives by combined score
6. Present the best option with improvement metrics

## Scalability Considerations

The architecture is designed to scale in several dimensions:

1. **Model Types**: New model types can be added to the `ModelFactory` class
2. **KPI Types**: Additional KPIs can be added by extending the `kpi_types` list
3. **Feature Engineering**: New features can be added to the feature engineering pipelines
4. **UI Components**: New visualizations can be added to the UI components

## Future Extensions

The architecture supports several potential extensions:

1. **Customer-Level Predictions**: Extending to predict individual customer behavior
2. **Personalization Models**: Adding models to predict optimal personalization strategies
3. **Automated A/B Testing**: Integrating with email service providers for automated testing
4. **Time Series Forecasting**: Adding models to predict KPI trends over time
5. **Natural Language Understanding**: Deeper analysis of subject line semantics