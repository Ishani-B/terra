import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess eye-tracking data
def load_data(folder, label):
    """Load CSV feature data and assign labels."""
    data = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            data.append(df.values[0])  # Extract first row as feature vector
    data = np.array(data)
    labels = np.full((data.shape[0], 1), label)
    return np.hstack((data, labels))

# Load healthy & Alzheimer's eye-tracking data
healthy_data = load_data("healthy_eye", 0)
ad_data = load_data("ad_eye", 1)
data = np.vstack((healthy_data, ad_data))

# Split into features & labels
X = data[:, :-1]
y = data[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train SVM
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(svm_model, "svm_model.pkl")

# Evaluate
y_pred = svm_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("✅ SVM Model Saved!")
