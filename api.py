# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Network Anomaly Detection API")

# 1. Load Models (Global Scope)
try:
    pipeline = joblib.load('../artifacts/preprocessing_pipeline.joblib')
    autoencoder = load_model('../artifacts/autoencoder_model.keras')
    rf_model = joblib.load('../artifacts/random_forest_model.joblib')
    print("Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# 2. Define Input Schema (Adjust fields based on your X_train columns)
class NetworkTraffic(BaseModel):
    Destination_Port: int
    Flow_Duration: float
    Total_Fwd_Packets: int
    Total_Backward_Packets: int
    # ... Add other key features here ...

# 3. Prediction Endpoint
@app.post("/predict")
def predict(traffic: NetworkTraffic):
    data = pd.DataFrame([traffic.dict()])
    
    # Preprocess
    data_clean = data.copy() # Add your specific cleaning steps here if needed
    X_processed = pipeline.transform(data_clean)
    
    # Autoencoder Feature
    reconstructions = autoencoder.predict(X_processed, verbose=0)
    reconstruction_error = np.mean(np.power(X_processed - reconstructions, 2), axis=1)
    
    # Random Forest Prediction
    X_rf = data_clean.copy()
    X_rf['reconstruction_error'] = reconstruction_error
    prediction = rf_model.predict(X_rf)[0]
    probability = rf_model.predict_proba(X_rf)[0].max()
    
    return {
        "prediction": "Bot/Anomaly" if prediction == 1 else "Benign",
        "confidence": float(probability),
        "reconstruction_error": float(reconstruction_error[0])
    }

# 4. Instrument with Prometheus
Instrumentator().instrument(app).expose(app)