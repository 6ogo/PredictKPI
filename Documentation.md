# Sendout KPI Predictor Documentation

This directory contains documentation for the Sendout KPI Predictor models and their performance. The goal of this documentation is to ensure transparency, reproducibility, and maintainability of the prediction system.

## Model Versioning

Each model is versioned following semantic versioning (Major.Minor.Patch):

- **Major**: Breaking changes in feature structure or prediction interface
- **Minor**: Non-breaking additions of features or functionality
- **Patch**: Bug fixes and minor improvements

### Version History

- **1.0.0**: Initial model using only subject line features
- **2.0.0**: Added preheader features for improved prediction accuracy

## Model Features

### Legacy Features (v1.x.x)

The legacy model uses the following features:

1. **Categorical Features** (one-hot encoded):
   - Dialog
   - Syfte (Purpose)
   - Product

2. **Numerical Features**:
   - `Min_age`: Minimum age in the target audience
   - `Max_age`: Maximum age in the target audience
   - `Subject_length`: Character count in the subject line
   - `Num_words`: Word count in the subject line
   - `Has_exclamation`: 1 if subject contains '!', 0 otherwise
   - `Has_question`: 1 if subject contains '?', 0 otherwise

3. **Bolag Features** (one-hot encoded):
   - Binary indicators for each Bolag (Region/Company)

### V2 Features (v2.x.x)

Version 2.0.0 includes all legacy features plus preheader analysis:

1. **Categorical Features** (same as v1):
   - Dialog
   - Syfte (Purpose)
   - Product

2. **Numerical Features** (renamed + new):
   - `Min_age`, `Max_age`: Same as v1
   - `Subject_length`: Same as v1
   - `Subject_num_words`: Renamed from `Num_words`
   - `Subject_has_exclamation`: Renamed from `Has_exclamation`
   - `Subject_has_question`: Renamed from `Has_question`
   - `Preheader_length`: Character count in preheader
   - `Preheader_num_words`: Word count in preheader
   - `Preheader_has_exclamation`: 1 if preheader contains '!', 0 otherwise
   - `Preheader_has_question`: 1 if preheader contains '?', 0 otherwise

3. **Bolag Features** (same as v1)

## Feature Engineering

Features are engineered from two primary data sources:

1. **Delivery Data**: Contains information about each email campaign
   - Subject lines, preheaders
   - Campaign metadata (Dialog, Syfte, Product)
   - Performance metrics (Opens, Clicks, Optouts, Sendouts)

2. **Customer Data**: Contains information about the target audience
   - Age demographics
   - Regional distribution (Bolag)

### Preprocessing Steps

1. **Calculate KPIs**:
   - `Openrate` = Opens / Sendouts
   - `Clickrate` = Clicks / Sendouts
   - `Optoutrate` = Optouts / Sendouts

2. **Aggregate Age Stats**:
   - Group by `InternalName`
   - Calculate min and max age for each group

3. **Create Bolag Features**:
   - One-hot encode the `Bolag` column
   - Aggregate by `InternalName`

4. **Extract Text Features**:
   - Calculate length and word count
   - Check for exclamation and question marks

## Model Performance

Each model version has its own performance metrics documented in:
- `model_vX.X.X/model_documentation.yaml`
- `model_vX.X.X/feature_importances.csv`

Performance metrics include:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Usage Guidelines

### Choosing the Right Model Version

- Use v2.x.x models when preheader information is available
- Use v1.x.x models when only subject line data is available

### Feature Compatibility

When using the application:
1. The app automatically detects the model version
2. It validates feature compatibility
3. If mismatches are detected, it adapts the input data to match model expectations

### Prediction Interface

Inputs required for predictions:
- Dialog, Syfte, Product selections
- Target audience information (age range, included Bolag)
- Subject line text
- Preheader text (for v2.x.x models)

## Implementation Details

### XGBoost Model

The prediction model uses XGBoost regression with the following key parameters:
- L2 regularization (reg_lambda=1.0)
- Sample weighting to emphasize high-performing emails

### Training Strategy

- 80/20 train/test split
- 5-fold cross-validation on training data
- Higher weights for emails with >50% open rates

## Maintenance

### Adding New Features

When adding new features:
1. Increment the major version number
2. Document the new features in this README
3. Create a new model file with the updated version number
4. Update the feature engineering function to include the new features

### Retraining Existing Models

To retrain an existing model:
1. Use the "Force model retraining" option in the application
2. The existing model will be overwritten with new weights
3. Performance metrics will be updated

## Troubleshooting

Common issues and solutions:

1. **Feature Mismatch Errors**:
   - Check that feature names match between training and prediction
   - Use the `adapt_features_to_model` function to align features

2. **Missing Categorical Values**:
   - New Dialog/Syfte/Product values will not have corresponding columns in the model
   - Use the closest available category or retrain the model

3. **Performance Degradation**:
   - Check for data distribution shifts
   - Evaluate feature importance changes
   - Consider retraining with more recent data

## Contact

For any questions about this documentation or the model, please contact the data science team.