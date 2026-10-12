import re
import numpy as np
import pandas as pd

# Matches feature names like "sensor_turbidity (t-3)" -> ("sensor_turbidity", 3)
LAG_PATTERN = re.compile(r"^(?P<base>.+?)\s*\(t-(?P<lag>\d+)\)$")


def base_name(col):
    """'sensor_turbidity (t-3)' -> 'sensor_turbidity'. Returns col unchanged if no lag suffix."""
    m = LAG_PATTERN.match(col)
    return m.group("base") if m else col


def select_features(data, feature_cols):
    """Select model features in training order; fail loudly instead of silently filling NaN."""
    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} model feature(s) missing from input, e.g. {missing[:5]}. "
            f"Input columns look like: {list(data.columns[:5])}"
        )
    return data[feature_cols]


def scale_features(df, scaler_dict):
    """
    Apply per-variable scalers to every lag column of that variable.
    Works whether scaler_dict is keyed by full column name ('sensor_turbidity (t-3)')
    or by base variable name ('sensor_turbidity').
    """
    df_scaled = df.astype(float).copy()
    unscaled = []
    for col in df.columns:
        scaler = scaler_dict.get(col, scaler_dict.get(base_name(col)))
        if scaler is None:
            unscaled.append(col)
            continue
        # Rename the column to the name the scaler was fitted with (if any), so sklearn doesn't complain
        fitted_names = getattr(scaler, "feature_names_in_", None)
        x = df[[col]].astype(float)
        x = x.set_axis(list(fitted_names), axis=1) if fitted_names is not None else x.to_numpy()
        df_scaled[col] = scaler.transform(x).ravel()
    if unscaled:
        bases = sorted({base_name(c) for c in unscaled})
        print(f"[i] No scaler found for: {bases} (left unscaled)")
    return df_scaled


def run_classification_model(data, model, scaler_dict, feature_cols):
    print("\n[Running Classification Model...]")

    df = select_features(data, feature_cols)
    df_scaled = scale_features(df, scaler_dict)

    prediction = model.predict(df_scaled)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_scaled)[0]  # was: unscaled df
        confidence = max(probabilities) * 100
        return f"Class {prediction} (Confidence: {confidence:.2f}%)"

    return f"Class Prediction: {prediction}"


def run_regression_model(data, model, scaler_dict, feature_cols, horizon_name):
    print(f"\n[Running Regression Model for {horizon_name} Horizon...]")

    df = select_features(data, feature_cols)
    df_scaled = scale_features(df, scaler_dict)

    prediction_scaled = np.asarray(model.predict(df_scaled)[0]).reshape(-1, 1)

    # Convert the scaled prediction back to real turbidity units
    prediction = scaler_dict["sensor_turbidity"].inverse_transform(prediction_scaled).ravel()
    return f"Predicted Value for {horizon_name}: {prediction[0]:.4f}"