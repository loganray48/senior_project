# Senior Project: Car Regression ML Prediction

## Project Overview

This competition provides a synthetically-generated dataset of used cars, and the goal is to **predict the price of each car** based on various attributes.

## 🔍 Goal

Predict car prices from structured tabular data using modern machine learning models, stacking techniques, and feature engineering.

## Techniques Used

This repository includes the following:

- Feature engineering:
  - Outlier classification using `CatBoostClassifier`
  - Median target encoding (leak-free)
  - Label encoding with rare class grouping
    
- Models:
  - XGBoost (GPU)
  - LightGBM (GPU)
  - Support Vector Regression (RBF Kernel)
  - AutoGluon FastAI Tabular Model
    
- Stacked model:
  - Ridge regression on out-of-fold predictions


## Directory Structure 
