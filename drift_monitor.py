# src/drift_monitor.py
import pandas as pd
from scipy.stats import ks_2samp
import joblib

# Load reference data (from training)
X_train = pd.read_csv("../artifacts/train_sample.csv") # You need to save a sample during training!

# Load new production data (collected from logs)
X_prod = pd.read_csv("production_logs.csv") 

def check_drift(reference, current, threshold=0.05):
    drifted_features = []
    for column in reference.columns:
        # Kolmogorov-Smirnov Test
        stat, p_value = ks_2samp(reference[column], current[column])
        if p_value < threshold:
            drifted_features.append((column, p_value))
    return drifted_features

drifts = check_drift(X_train, X_prod)
if drifts:
    print(f"ALERT: Data drift detected in features: {drifts}")
else:
    print("System healthy. No drift detected.")