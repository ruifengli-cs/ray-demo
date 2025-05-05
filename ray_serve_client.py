
import requests
import numpy as np
from sklearn.datasets import load_breast_cancer
import time
import json

# Load a sample data point
X, y = load_breast_cancer(return_X_y=True)

# Function to make a prediction request
def make_prediction(sample_index=0):
    sample = X[sample_index].tolist()

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"data": sample}
        )
        result = response.json()
        print(f"Prediction result: {result}")
        print(f"True label: {y[sample_index]}")
        return result
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Make multiple predictions with different samples
print("Making predictions...")
for i in range(10):
    print(f"\nSample {i}:")
    make_prediction(i)
    time.sleep(2)  # Wait 2 seconds between requests
