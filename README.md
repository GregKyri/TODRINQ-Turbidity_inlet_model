# Turbidity Prediction Project

This repository contains 3 XGBoost regression models for predicting turbidity with a predictive horizon of 1,3, and 6 hours ahead. 
Moreover, it also contains a Random Forest classification model for the prediction of turbidity events (3 consecutive hours above 20FTU).

---

## Requirements

- Python >= 3.9
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

