# Turbidity Prediction Project

This repository contains 1 XGBoost regression model and one Random forest model for predicting turbidity with a predictive horizon of 3 and 6 hours ahead respectively. 
An XGBoost and Random Forest classification model for the prediction of turbidity events (3 consecutive hours above 20FTU) for a predictive horizon of 3 and 6 hours ahead respectively.

---

## Requirements

- Python >= 3.1
- Packages: captum
numpy
pandas
plotly
seaborn
sklearn
scipy
json
os
joblib

---

## Predicting on New Data

Open `make_predictions.py` and run the script.

New data shoud contain: 24hour timelaged i) turbidity in the inlet ii) temperature, iii) Hagenhein discharge and iv) month data
Total number of initial variables 96

## Training and Testing Scripts

All training and evaluations scripts are in the Training Scripts folder. 
**Main script.ipynb** is the script that runs training of regression models, SHAP analysis, classification models and Boostrap analysis.
**new_version_training.py** script for training LSTM,ARIMA,RF and XGBoost models
**Boostrap.py** script for the boostrap uncertainty analysis
**bootstrap_horizons** produces plots over 6 hours ahead of the boostrap analysis
**shap_feature_importance.py** script that runs the SHAP analysis and creates block feature importance (sum of 24 hours timelagged data) per different variable 
**new_version_testing.ipynb** script reads the results of the train models and produces comparison metrics (MAE,RMSE,NSE,KGE, P-bias, SS and 90th percentile MAE)
**Event_classification** script for the classification approach

## Best models

The Best models forlder contains the best trained models per predictive horizon (3 and 6 hours ahead) and predictive purposes (regression - classification).


