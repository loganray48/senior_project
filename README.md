# Senior Project: Car Regression ML Prediction

## Project Overview

This senior project focuses on predicting used car prices using a synthetically generated dataset provided by Kaggle's Tabular Playground Series. The goal is to build accurate regression models using structured tabular data enhanced with advanced feature engineering and model stacking techniques.

## Goal

Predict car prices based on vehicle attributes such as brand, fuel type, mileage, model year, and transmission type. The project includes data cleaning, exploratory analysis, model training, evaluation, and dashboard development.

---

## Techniques Used

### Feature Engineering & Data Augmentation

- **New columns created:**
  - `car_age` (based on model year)
  - `mileage_per_year` (adjusted for age)
  - `is_luxury` (brand flag for luxury vehicles)
  - `transmission_simplified` (collapsed A/T vs. M/T)
- **Feature combinations:** including `brand_model`, `brand_mileage`, `brand_ext_col`, and `brand_int_col`
- **Outlier classification:** Used IQR filtering and trained a `CatBoostClassifier` to flag extreme values
- **Target encoding:** Leak-free median target encoding for categorical features
- **Meta-features from models:** Out-of-fold predictions from XGBoost, LightGBM, and SVR were used as input features for a stacked model
- **AutoML predictions:** Integrated features and results from AutoGluon’s FastAI model

### Models

- XGBoost Regressor (with GPU)
- LightGBM Regressor (with GPU)
- Support Vector Regressor (RBF kernel)
- AutoGluon TabularPredictor (FastAI backend)
- Ridge Regression for final ensemble stacking

---

## Streamlit Dashboard

An interactive dashboard built with Streamlit includes:

- Exploratory data visualizations
- Data validation and preprocessing overview
- Model performance metrics and comparisons
- Feature importance charts
- Predicted vs. actual price scatter plots

To run the app locally:

```bash
cd Streamlit
streamlit run app.py
```

senior_project/
├── AutogluonModels/           # Output from AutoGluon training
├── Streamlit/                 # Streamlit dashboard application
│   └── app.py
├── catboost_info/             # CatBoost logs
├── data/                      # Raw CSV and zipped datasets
├── charts.py                  # Plotly and seaborn visualizations
├── project.py                 # Main ML pipeline and modeling logic
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation

