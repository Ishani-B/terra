import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for healthy individuals
healthy_params = {
    "latency_mean": 200,  # Mean latency (ms)
    "latency_std": 30,    # Standard deviation for latency
    "gain_mean": 1.0,     # Mean gain (normalized)
    "gain_std": 0.1,      # Standard deviation for gain
    "errors_mean": 1,     # Mean number of errors
    "errors_std": 0.5,    # Standard deviation for errors
    "rate_mean": 3,       # Mean saccade rate (saccades/second)
    "rate_std": 0.5       # Standard deviation for rate
}

# Parameters for Alzheimer's patients (based on the table)
ad_params = {
    "latency_mean": 400,  # Greatly increased latency (ms)
    "latency_std": 50,    # Standard deviation for latency
    "gain_mean": 1.0,     # Normal gain (no change)
    "gain_std": 0.1,      # Standard deviation for gain
    "errors_mean": 5,     # Increased errors
    "errors_std": 1,      # Standard deviation for errors
    "rate_mean": 1,       # Greatly decreased saccade rate (saccades/second)
    "rate_std": 0.3       # Standard deviation for rate
}

# Generate synthetic data for healthy individuals
num_subjects = 50
healthy_data = {
    "latency": np.random.normal(healthy_params["latency_mean"], healthy_params["latency_std"], num_subjects),
    "gain": np.random.normal(healthy_params["gain_mean"], healthy_params["gain_std"], num_subjects),
    "errors": np.random.normal(healthy_params["errors_mean"], healthy_params["errors_std"], num_subjects),
    "rate": np.random.normal(healthy_params["rate_mean"], healthy_params["rate_std"], num_subjects)
}

# Generate synthetic data for Alzheimer's patients
ad_data = {
    "latency": np.random.normal(ad_params["latency_mean"], ad_params["latency_std"], num_subjects),
    "gain": np.random.normal(ad_params["gain_mean"], ad_params["gain_std"], num_subjects),
    "errors": np.random.normal(ad_params["errors_mean"], ad_params["errors_std"], num_subjects),
    "rate": np.random.normal(ad_params["rate_mean"], ad_params["rate_std"], num_subjects)
}

# Create folders if they don't exist
os.makedirs("healthy_eye", exist_ok=True)
os.makedirs("ad_eye", exist_ok=True)

# Save healthy data to CSV files
for i in range(num_subjects):
    healthy_case = {
        "latency": healthy_data["latency"][i],
        "gain": healthy_data["gain"][i],
        "errors": healthy_data["errors"][i],
        "rate": healthy_data["rate"][i]
    }
    df = pd.DataFrame([healthy_case])
    df.to_csv(f"healthy_eye/healthy_case_{i+1}.csv", index=False)

# Save Alzheimer's data to CSV files
for i in range(num_subjects):
    ad_case = {
        "latency": ad_data["latency"][i],
        "gain": ad_data["gain"][i],
        "errors": ad_data["errors"][i],
        "rate": ad_data["rate"][i]
    }
    df = pd.DataFrame([ad_case])
    df.to_csv(f"ad_eye/ad_case_{i+1}.csv", index=False)

print("CSV files generated successfully!")
 