import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Create model directory if it doesn't exist
MODEL_DIR = r"C:\Users\naren\Desktop\Python codes project\fastapi\model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset
print("Loading dataset...")
data_path = r"C:\Users\naren\Desktop\Python codes project\Untitled Folder\dataset.csv"
df = pd.read_csv(data_path)

# Define feature and target columns
feature_cols = ['DO', 'pH', 'Alkalinity', 'Hardness', 'Nitrite', 'H2S', 'Salinity', 'Ammonia', 'Temperature']
target_cols = ['WSS', 'AHPND', 'TSV', 'YHV']

# Split features and targets
X = df[feature_cols]
y = df[target_cols]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Scaler saved successfully.")

# Train a Random Forest model for each disease
models = {}
results = {}

for disease in target_cols:
    print(f"Training Random Forest model for {disease}...")
    
    # Create and train the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train[disease])
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test[disease], y_pred)
    report = classification_report(y_test[disease], y_pred)
    
    print(f"Accuracy for {disease}: {accuracy:.4f}")
    print(f"Classification Report for {disease}:\n{report}")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"rf_model_{disease}.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Model for {disease} saved to {model_path}")
    
    # Store model and results
    models[disease] = rf_model
    results[disease] = {
        'accuracy': accuracy,
        'report': report
    }

# Save metadata
metadata = {
    'feature_columns': feature_cols,
    'target_columns': target_cols,
    'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.pkl"))
print("Metadata saved successfully.")

print("\nAll models trained and saved successfully!")
print(f"Models are saved in: {MODEL_DIR}")