from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ── Load models & artifacts ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logistic_model   = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
isolation_forest = joblib.load(os.path.join(BASE_DIR, "models", "isolation_forest_model.pkl"))
scaler           = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

with open(os.path.join(BASE_DIR, "models", "feature_columns.json")) as f:
    feature_columns = json.load(f)

# ── Feature metadata for the UI form (user-facing fields) ────────────────────
# These are the RAW fields the user fills in. We one-hot encode them before
# passing to the model (which expects the one-hot columns in feature_columns.json).
FEATURE_META = {
    "age":                     {"label": "Age",                              "min": 29,  "max": 77,  "step": 1,   "type": "number"},
    "gender":                  {"label": "Sex",                              "min": 0,   "max": 1,   "step": 1,   "type": "select", "options": {"0": "Female", "1": "Male"}},
    "chest_pain_type":         {"label": "Chest Pain Type",                  "min": 0,   "max": 3,   "step": 1,   "type": "select", "options": {"0": "Typical Angina", "1": "Atypical Angina", "2": "Non-Anginal Pain", "3": "Asymptomatic"}},
    "resting_blood_pressure":  {"label": "Resting Blood Pressure (mmHg)",    "min": 94,  "max": 200, "step": 1,   "type": "number"},
    "serum_cholesterol":       {"label": "Serum Cholesterol (mg/dl)",        "min": 126, "max": 564, "step": 1,   "type": "number"},
    "fasting_blood_sugar":     {"label": "Fasting Blood Sugar > 120 mg/dl", "min": 0,   "max": 1,   "step": 1,   "type": "select", "options": {"0": "No", "1": "Yes"}},
    "resting_ecg_results":     {"label": "Resting ECG Results",              "min": 0,   "max": 2,   "step": 1,   "type": "select", "options": {"0": "Normal", "1": "ST-T Wave Abnormality", "2": "Left Ventricular Hypertrophy"}},
    "maximum_heart_rate":      {"label": "Maximum Heart Rate Achieved",      "min": 71,  "max": 202, "step": 1,   "type": "number"},
    "exercise_induced_angina": {"label": "Exercise Induced Angina",          "min": 0,   "max": 1,   "step": 1,   "type": "select", "options": {"0": "No", "1": "Yes"}},
    "st_depression":           {"label": "ST Depression (Oldpeak)",          "min": 0.0, "max": 6.2, "step": 0.1, "type": "number"},
    "st_segment_slope":        {"label": "Slope of Peak Exercise ST",        "min": 0,   "max": 2,   "step": 1,   "type": "select", "options": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"}},
    "number_of_major_vessels": {"label": "Number of Major Vessels (0-4)",    "min": 0,   "max": 4,   "step": 1,   "type": "select", "options": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}},
    "thalassemia_type":        {"label": "Thalassemia Type",                 "min": 0,   "max": 3,   "step": 1,   "type": "select", "options": {"0": "None", "1": "Fixed Defect", "2": "Normal", "3": "Reversible Defect"}},
}

# Ordered list of user-facing fields (what the HTML form renders)
FORM_FIELDS = [
    "age", "gender", "chest_pain_type", "resting_blood_pressure",
    "serum_cholesterol", "fasting_blood_sugar", "resting_ecg_results",
    "maximum_heart_rate", "exercise_induced_angina", "st_depression",
    "st_segment_slope", "number_of_major_vessels", "thalassemia_type",
]

# Categorical fields that were one-hot encoded during training
ONE_HOT_CATS = ["chest_pain_type", "resting_ecg_results", "st_segment_slope", "thalassemia_type"]


# ── One-hot encoder ───────────────────────────────────────────────────────────
def encode_input(data: dict) -> pd.DataFrame:
    """
    Converts raw form data (categorical values as integers) into a one-hot
    encoded DataFrame matching feature_columns.json exactly.
    """
    # Start with all columns zeroed out
    row = {col: 0.0 for col in feature_columns}

    # Copy scalar / binary fields directly
    for field in FORM_FIELDS:
        if field in ONE_HOT_CATS:
            continue
        val = data.get(field)
        if val is None:
            raise ValueError(f"Missing field: {field}")
        if field in row:
            row[field] = float(val)

    # One-hot encode categorical fields
    for cat in ONE_HOT_CATS:
        val = data.get(cat)
        if val is None:
            raise ValueError(f"Missing field: {cat}")
        level = int(float(val))
        ohe_col = f"{cat}_{level}"
        if ohe_col in row:   # level 0 = reference category, stays 0
            row[ohe_col] = 1.0

    return pd.DataFrame([row], columns=feature_columns)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        features=FEATURE_META,
        feature_columns=FORM_FIELDS,  # JS uses this to build the form
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify({"features": FEATURE_META, "feature_columns": FORM_FIELDS})


@app.route("/api/predict/logistic", methods=["POST"])
def predict_logistic():
    try:
        data       = request.get_json(force=True)
        df_input   = encode_input(data)
        scaled     = scaler.transform(df_input)
        prediction = int(logistic_model.predict(scaled)[0])
        proba      = logistic_model.predict_proba(scaled)[0].tolist()
        risk_pct   = round(proba[1] * 100, 2)
        risk_level = "Low" if risk_pct < 35 else "Medium" if risk_pct < 65 else "High"

        return jsonify({
            "model":       "Logistic Regression",
            "prediction":  prediction,
            "label":       "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "probability": {"no_disease": round(proba[0] * 100, 2), "disease": risk_pct},
            "risk_level":  risk_level,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


@app.route("/api/predict/isolation-forest", methods=["POST"])
def predict_isolation_forest():
    try:
        data        = request.get_json(force=True)
        df_input    = encode_input(data)
        scaled      = scaler.transform(df_input)
        raw_pred    = int(isolation_forest.predict(scaled)[0])
        score       = float(isolation_forest.decision_function(scaled)[0])
        is_anomaly  = raw_pred == -1
        anomaly_pct = round(max(0, min(100, (0.5 - score) * 100)), 2)

        return jsonify({
            "model":          "Isolation Forest",
            "raw_label":      raw_pred,
            "is_anomaly":     is_anomaly,
            "label":          "Anomalous Pattern Detected" if is_anomaly else "Normal Pattern",
            "anomaly_score":  anomaly_pct,
            "decision_score": round(score, 4),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


@app.route("/api/predict/combined", methods=["POST"])
def predict_combined():
    try:
        data        = request.get_json(force=True)
        df_input    = encode_input(data)
        scaled      = scaler.transform(df_input)

        lr_pred     = int(logistic_model.predict(scaled)[0])
        lr_proba    = logistic_model.predict_proba(scaled)[0].tolist()
        risk_pct    = round(lr_proba[1] * 100, 2)

        if_raw      = int(isolation_forest.predict(scaled)[0])
        if_score    = float(isolation_forest.decision_function(scaled)[0])
        is_anomaly  = if_raw == -1
        anomaly_pct = round(max(0, min(100, (0.5 - if_score) * 100)), 2)

        if lr_pred == 1 and is_anomaly:
            verdict, verdict_class = "High Risk", "danger"
        elif lr_pred == 1 or is_anomaly:
            verdict, verdict_class = "Moderate Risk", "warning"
        else:
            verdict, verdict_class = "Low Risk", "success"

        return jsonify({
            "logistic_regression": {
                "prediction": lr_pred,
                "label":      "Heart Disease Detected" if lr_pred == 1 else "No Heart Disease",
                "probability": {"no_disease": round(lr_proba[0] * 100, 2), "disease": risk_pct},
            },
            "isolation_forest": {
                "is_anomaly":    is_anomaly,
                "label":         "Anomalous Pattern" if is_anomaly else "Normal Pattern",
                "anomaly_score": anomaly_pct,
            },
            "verdict":       verdict,
            "verdict_class": verdict_class,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)