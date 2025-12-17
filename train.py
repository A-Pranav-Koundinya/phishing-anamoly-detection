# --- Part 1: Train the Autoencoder on 'Normal' Data ---
from sklearn.ensemble import RandomForestClassifier
import mlflow
print("Step 1: Training Autoencoder on 'BENIGN' (Monday) data...")

# Get the number of features from our processed training data
input_dim = X_train_processed.shape[1]
encoding_dim = 14 # Bottleneck size

# Define the Model Architecture
input_layer = Input(shape=(input_dim, ))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dense(input_dim, activation='linear')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
mlflow.set_experiment("Network_Anomaly_Detection")

with mlflow.start_run():
    # Log Parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("encoding_dim", 14)
# Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True)

    autoencoder.fit(X_train_processed, X_train_processed,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopper],
            verbose=1)
    mlflow.log_metric("accuracy", 0.98) # Replace with actual variable
    
    # Log Models
    mlflow.sklearn.log_model(rf_model, "random_forest")
    mlflow.keras.log_model(autoencoder, "autoencoder")

print("Autoencoder training complete.")

# --- Part 2: Use Autoencoder to Create New Feature ---

print("\nStep 2: Generating 'reconstruction_error' feature for 'Friday' data...")

# 1. Get the model's reconstructions of the test data
reconstructions = autoencoder.predict(X_test_processed)

# 2. Calculate the Mean Squared Error (MSE) for each sample
# THIS IS OUR NEW FEATURE
reconstruction_error = np.mean(np.power(X_test_processed - reconstructions, 2), axis=1)

# 3. Create a new "features" dataframe for the Random Forest
#    We use X_test (the *un*-scaled data) for better feature interpretability
#    and add our new error feature.
X_test_new_features = X_test.copy()
X_test_new_features['reconstruction_error'] = reconstruction_error

print("New feature added successfully.")

# --- Part 3: Train a Supervised Classifier ---

print("\nStep 3: Training Random Forest Classifier on new feature set...")

# We will train the Random Forest on the Friday data (which has labels)
# We can't use X_test_new_features for both train and test, so we must
# split the Friday data.
#
# IMPORTANT: We need to use y_test here, which has the *labels*
# We'll train and test *only* on the Friday file.

from sklearn.model_selection import train_test_split

# Split the Friday data into a new train/test set (e.g., 70% train, 30% test)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_test_new_features,
    y_test, # The original 'BENIGN'/'Bot' labels
    test_size=0.3,
    random_state=42,
    stratify=y_test # Ensures both train/test have proportional 'Bot' samples
)

print(f"Random Forest Training samples: {X_train_rf.shape[0]}")
print(f"Random Forest Testing samples: {X_test_rf.shape[0]}")

# Initialize the Random Forest Classifier
# class_weight='balanced' is CRITICAL: it tells the model to pay
# extra attention to the rare 'Bot' class.
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Train the model
rf_model.fit(X_train_rf, y_train_rf)

print("Random Forest training complete.")

# --- Part 4: Evaluate the Hybrid Model ---

print("\n--- Model Evaluation Results (Hybrid Model) ---")

# Make predictions on the 30% held-out test set
y_pred_rf = rf_model.predict(X_test_rf)

# Show the classification report
print(classification_report(y_test_rf, y_pred_rf))

# Plot the confusion matrix
cm = confusion_matrix(y_test_rf, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Hybrid Model (AE + RF) Confusion Matrix')
plt.show()

# --- Optional: Show Feature Importance ---
print("\nTop 10 Most Important Features:")
features = X_test_new_features.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10

for i in indices:
    print(f"{features[i]}: {importances[i]:.4f}")

# --- Part 5: Save the Models ---
import joblib

print("\nSaving models...")

# Save the Autoencoder model
autoencoder_path = 'autoencoder_model.keras'
autoencoder.save(autoencoder_path)
print(f"Autoencoder model saved to {autoencoder_path}")

# Save the Random Forest model
rf_model_path = 'random_forest_model.joblib'
joblib.dump(rf_model, rf_model_path)
print(f"Random Forest model saved to {rf_model_path}")

print("Models saved successfully. You can now download them from the Colab file browser (left sidebar -> folder icon).")