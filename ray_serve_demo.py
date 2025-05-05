# ray_serve_demo.py
# This demo shows how to train a model and serve it for inference simultaneously using Ray and Ray Serve

import os
import time
import ray
import numpy as np
import requests
from ray import serve
from ray.serve.deployment import Deployment
from ray.serve._private.common import DeploymentStatus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import threading
import json

# Initialize Ray - this will start a Ray cluster on your local machine
ray.init()


# Load a sample dataset (breast cancer dataset)
def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Define a model training function that will run as a Ray task
@ray.remote
def train_model(iteration):
    print(f"Starting training iteration {iteration}...")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train a simple model (we'll use RandomForest for demonstration)
    model = RandomForestClassifier(
        n_estimators=1 + (iteration * 5),  # Increase complexity with each iteration
        random_state=42
    )

    # Simulate longer training time
    time.sleep(5)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Completed training iteration {iteration}. Accuracy: {accuracy:.4f}")

    return model, accuracy


# Define a predictor class for Ray Serve
@serve.deployment(name="predictor")
class ModelPredictor:
    def __init__(self):
        self.model = None
        self.model_version = 0
        self.accuracy = 0
        self.ready = False
        # Load initial model
        X, y = load_breast_cancer(return_X_y=True)
        self.model = RandomForestClassifier(n_estimators=1, random_state=42)
        self.model.fit(X[:100], y[:100])  # Train on a small subset for initial model
        
        # Calculate initial accuracy
        y_pred = self.model.predict(X[100:])  # Predict on the rest of the data
        self.accuracy = accuracy_score(y[100:], y_pred)
        print(f"Initial model accuracy: {self.accuracy:.4f}")
        
        self.ready = True

    async def update_model(self, model, accuracy, version):
        print(f"Updating model to version {version} with accuracy {accuracy:.4f}")
        self.model = model
        self.accuracy = accuracy
        self.model_version = version

    async def __call__(self, request):
        if not self.ready or self.model is None:
            return {"error": "Model not ready yet"}

        # Parse the request data
        try:
            input_data = await request.json()
            data = np.array(input_data["data"]).reshape(1, -1)
        except Exception as e:
            return {"error": str(e)}

        # Make prediction
        try:
            prediction = int(self.model.predict(data)[0])
            probability = float(self.model.predict_proba(data)[0][1])

            return {
                "prediction": prediction,
                "probability": probability,
                "model_version": self.model_version,
                "model_accuracy": float(self.accuracy)
            }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

    async def status(self, request):
        return {
            "ready": self.ready,
            "model_version": self.model_version,
            "accuracy": float(self.accuracy)
        }


# Function to run the demo
async def run_demo():
    print("Starting Ray Serve demo: Training and inference happening simultaneously")

    # Start Ray Serve
    serve.start()

    # Deploy the predictor
    predictor = serve.run(ModelPredictor.bind(), route_prefix="/predict")
    print("Model predictor deployed")

    # Wait for deployment to be ready
    while True:
        status = serve.status()
        if status.applications["default"].deployments["predictor"].status == DeploymentStatus.HEALTHY:
            break
        print("Waiting for deployment to be ready...")
        time.sleep(1)

    print("Predictor is ready to serve requests")

    # Function to make a sample prediction request
    def make_prediction():
         # Load a sample data point
        X, _ = load_breast_cancer(return_X_y=True)
        sample = X[0].tolist()

        # Load data and find a malignant sample
        # X, y = load_breast_cancer(return_X_y=True)
        # # Find the first malignant sample (where y == 1)
        # malignant_indices = np.where(y == 1)[0]
        # sample_index = malignant_indices[0]  # Get the first malignant sample
        # sample = X[sample_index].tolist()

        # Make a prediction request
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"data": sample}
            )
            result = response.json()
            print(f"Prediction result: {result}")
            # print(f"True label: {y[sample_index]}")  # Print the true label
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    # Run inference in a loop
    def inference_loop():
        print("Starting inference loop...")
        for i in range(20):
            print(f"\nInference request {i + 1}:")
            result = make_prediction()
            time.sleep(3)  # Make a request every 3 seconds

        print("Inference loop complete!")

    # Start inference in a separate thread
    inference_thread = threading.Thread(target=inference_loop)
    inference_thread.daemon = True
    inference_thread.start()

    # Start training in the background and update the model as training progresses
    model_futures = []
    for i in range(5):  # Train 5 iterations of the model
        # Start a training task
        model_future = train_model.remote(i)
        model_futures.append(model_future)

        # Wait for this training iteration to complete
        model, accuracy = ray.get(model_future)

        # Get a handle to the deployment and update the model
        handle = serve.get_deployment_handle("predictor", app_name="default")
        await handle.update_model.remote(model, accuracy, i + 1)

        # Small delay between training iterations
        if i < 4:  # Don't sleep after the last iteration
            time.sleep(8)

    # Wait for the inference thread to complete
    inference_thread.join()

    print("Demo completed!")

    # Shutdown Ray Serve
    serve.shutdown()

    # Shutdown Ray
    ray.shutdown()


# Define a simple client script to test the model service
def create_client_script():
    with open("ray_serve_client.py", "w") as f:
        f.write("""
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
    print(f"\\nSample {i}:")
    make_prediction(i)
    time.sleep(2)  # Wait 2 seconds between requests
""")


if __name__ == "__main__":
    # Create the client script for later use
    create_client_script()

    # Run the demo
    import asyncio
    asyncio.run(run_demo())