# Heart Disease Prediction System

A data mining project that predicts heart disease risk using two complementary ML approaches: **Logistic Regression** (supervised classification) and **Isolation Forest** (anomaly detection). A Flask web app provides a clean UI and REST API for real-time predictions.

---

## Demo

The web app accepts 13 clinical features and returns:
- A **Logistic Regression** prediction with disease probability
- An **Isolation Forest** anomaly score
- A **combined verdict** (Low / Moderate / High Risk)

---

## Project Structure

```
heart-disease-predictor/
├── app.py                          # Flask application & REST API
├── requirements.txt                # Python dependencies
├── heart_disease_data.csv          # UCI Heart Disease dataset
│
├── models/
│   ├── feature_columns.json        # One-hot encoded feature schema
│   ├── logistic_model.pkl          # Trained Logistic Regression model
│   ├── isolation_forest_model.pkl  # Trained Isolation Forest model
│   └── scaler.pkl                  # StandardScaler fitted on training data
│
├── templates/
│   └── index.html                  # Frontend UI
│
└── notebooks/
    └── heart_disease_analysis.ipynb  # Full EDA, preprocessing & training pipeline
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/heart-disease-predictor.git
cd heart-disease-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models (optional)

The pre-trained `.pkl` files are included. To retrain from scratch, open and run the notebook:

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

> The notebook saves models to `models/`.

### 3. Run the web app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Reference

All endpoints accept and return JSON.

### `GET /api/health`
Health check.

```json
{ "status": "ok", "models_loaded": true }
```

---

### `GET /api/features`
Returns the full feature schema used by the UI form.

---

### `POST /api/predict/logistic`
Logistic Regression prediction.

**Request body:**
```json
{
  "age": 52,
  "gender": 1,
  "chest_pain_type": 0,
  "resting_blood_pressure": 125,
  "serum_cholesterol": 212,
  "fasting_blood_sugar": 0,
  "resting_ecg_results": 1,
  "maximum_heart_rate": 168,
  "exercise_induced_angina": 0,
  "st_depression": 1.0,
  "st_segment_slope": 2,
  "number_of_major_vessels": 2,
  "thalassemia_type": 3
}
```

**Response:**
```json
{
  "model": "Logistic Regression",
  "prediction": 1,
  "label": "Heart Disease Detected",
  "probability": { "no_disease": 23.4, "disease": 76.6 },
  "risk_level": "High"
}
```

---

### `POST /api/predict/isolation-forest`
Isolation Forest anomaly detection.

**Response:**
```json
{
  "model": "Isolation Forest",
  "is_anomaly": true,
  "label": "Anomalous Pattern Detected",
  "anomaly_score": 61.2,
  "decision_score": -0.112
}
```

---

### `POST /api/predict/combined`
Runs both models and returns a combined risk verdict.

**Response:**
```json
{
  "logistic_regression": { "prediction": 1, "label": "Heart Disease Detected", "probability": {...} },
  "isolation_forest":    { "is_anomaly": true, "label": "Anomalous Pattern", "anomaly_score": 61.2 },
  "verdict":             "High Risk",
  "verdict_class":       "danger"
}
```

**Verdict logic:**
| LR result | Isolation Forest | Verdict |
|-----------|-----------------|---------|
| Disease   | Anomaly         | High Risk |
| Disease **or** Anomaly (but not both) | — | Moderate Risk |
| No disease | Normal | Low Risk |

---

## Dataset

UCI Heart Disease dataset (`heart_disease_data.csv`) — 303 patient records, 13 clinical features.

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `gender` | 0 = Female, 1 = Male |
| `chest_pain_type` | 0–3 (Typical Angina → Asymptomatic) |
| `resting_blood_pressure` | mmHg |
| `serum_cholesterol` | mg/dl |
| `fasting_blood_sugar` | > 120 mg/dl: 1 = Yes |
| `resting_ecg_results` | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| `maximum_heart_rate` | bpm |
| `exercise_induced_angina` | 0 = No, 1 = Yes |
| `st_depression` | ST depression induced by exercise (oldpeak) |
| `st_segment_slope` | 0 = Up, 1 = Flat, 2 = Down |
| `number_of_major_vessels` | 0–4, coloured by fluoroscopy |
| `thalassemia_type` | 0 = None, 1 = Fixed Defect, 2 = Normal, 3 = Reversible Defect |

**Target:** `heart_disease` — 0 = No disease, 1 = Disease present

---

## ML Pipeline

See `notebooks/heart_disease_analysis.ipynb` for the full walkthrough:

1. **EDA** — distributions, correlation heatmap, class balance
2. **Preprocessing** — renaming columns, one-hot encoding categoricals, `StandardScaler`
3. **Logistic Regression** — trained with `sklearn`, evaluated on accuracy, precision, recall, F1, ROC-AUC
4. **Isolation Forest** — unsupervised anomaly detection, contamination tuned to dataset
5. **Model export** — `joblib` dumps for `logistic_model.pkl`, `isolation_forest_model.pkl`, `scaler.pkl`, `feature_columns.json`

---

## Tech Stack

| Layer | Library |
|-------|---------|
| ML | scikit-learn 1.5 |
| Data | pandas, numpy |
| API | Flask 3.0, flask-cors |
| Production server | Gunicorn |

---

## License

MIT
