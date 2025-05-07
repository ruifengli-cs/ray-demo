import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Ray Serve Demo\n",
                "\n",
                "This notebook demonstrates how to train a model and serve it for inference simultaneously using Ray and Ray Serve."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import os\n",
                "import time\n",
                "import ray\n",
                "import numpy as np\n",
                "import requests\n",
                "from ray import serve\n",
                "from ray.serve.deployment import Deployment\n",
                "from ray.serve._private.common import DeploymentStatus\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.datasets import load_breast_cancer\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score\n",
                "import threading\n",
                "import json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Initialize Ray and Load Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize Ray\n",
                "ray.init()\n",
                "\n",
                "# Load a sample dataset (breast cancer dataset)\n",
                "def load_data():\n",
                "    X, y = load_breast_cancer(return_X_y=True)\n",
                "    X_train, X_test, y_train, y_test = train_test_split(\n",
                "        X, y, test_size=0.2, random_state=42\n",
                "    )\n",
                "    return X_train, X_test, y_train, y_test"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Define Model Training Function"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "@ray.remote\n",
                "def train_model(iteration):\n",
                "    print(f\"Starting training iteration {iteration}...\")\n",
                "\n",
                "    # Load data\n",
                "    X_train, X_test, y_train, y_test = load_data()\n",
                "\n",
                "    # Train a simple model (we'll use RandomForest for demonstration)\n",
                "    model = RandomForestClassifier(\n",
                "        n_estimators=1 + (iteration * 5),  # Increase complexity with each iteration\n",
                "        random_state=42\n",
                "    )\n",
                "\n",
                "    # Simulate longer training time\n",
                "    time.sleep(5)\n",
                "\n",
                "    # Train the model\n",
                "    model.fit(X_train, y_train)\n",
                "\n",
                "    # Evaluate the model\n",
                "    y_pred = model.predict(X_test)\n",
                "    accuracy = accuracy_score(y_test, y_pred)\n",
                "\n",
                "    print(f\"Completed training iteration {iteration}. Accuracy: {accuracy:.4f}\")\n",
                "\n",
                "    return model, accuracy"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Define Model Predictor Class"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "@serve.deployment(name=\"predictor\")\n",
                "class ModelPredictor:\n",
                "    def __init__(self):\n",
                "        self.model = None\n",
                "        self.model_version = 0\n",
                "        self.accuracy = 0\n",
                "        self.ready = False\n",
                "        # Load initial model\n",
                "        X, y = load_breast_cancer(return_X_y=True)\n",
                "        self.model = RandomForestClassifier(n_estimators=1, random_state=42)\n",
                "        self.model.fit(X[:100], y[:100])  # Train on a small subset for initial model\n",
                "        \n",
                "        # Calculate initial accuracy\n",
                "        y_pred = self.model.predict(X[100:])  # Predict on the rest of the data\n",
                "        self.accuracy = accuracy_score(y[100:], y_pred)\n",
                "        print(f\"Initial model accuracy: {self.accuracy:.4f}\")\n",
                "        \n",
                "        self.ready = True\n",
                "\n",
                "    async def update_model(self, model, accuracy, version):\n",
                "        print(f\"Updating model to version {version} with accuracy {accuracy:.4f}\")\n",
                "        self.model = model\n",
                "        self.accuracy = accuracy\n",
                "        self.model_version = version\n",
                "\n",
                "    async def __call__(self, request):\n",
                "        if not self.ready or self.model is None:\n",
                "            return {\"error\": \"Model not ready yet\"}\n",
                "\n",
                "        # Parse the request data\n",
                "        try:\n",
                "            input_data = await request.json()\n",
                "            data = np.array(input_data[\"data\"]).reshape(1, -1)\n",
                "        except Exception as e:\n",
                "            return {\"error\": str(e)}\n",
                "\n",
                "        # Make prediction\n",
                "        try:\n",
                "            prediction = int(self.model.predict(data)[0])\n",
                "            probability = float(self.model.predict_proba(data)[0][1])\n",
                "\n",
                "            return {\n",
                "                \"prediction\": prediction,\n",
                "                \"probability\": probability,\n",
                "                \"model_version\": self.model_version,\n",
                "                \"model_accuracy\": float(self.accuracy)\n",
                "            }\n",
                "        except Exception as e:\n",
                "            return {\"error\": f\"Prediction error: {str(e)}\"}\n",
                "\n",
                "    async def status(self, request):\n",
                "        return {\n",
                "            \"ready\": self.ready,\n",
                "            \"model_version\": self.model_version,\n",
                "            \"accuracy\": float(self.accuracy)\n",
                "        }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Define Helper Functions for the Demo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def make_prediction():\n",
                "    # Load a sample data point\n",
                "    X, y = load_breast_cancer(return_X_y=True)\n",
                "    sample = X[0].tolist()\n",
                "\n",
                "    # Make a prediction request\n",
                "    try:\n",
                "        response = requests.post(\n",
                "            \"http://localhost:8000/predict\",\n",
                "            json={\"data\": sample}\n",
                "        )\n",
                "        result = response.json()\n",
                "        print(f\"Prediction result: {result}\")\n",
                "        print(f\"True label: {y[0]}\")\n",
                "        return result\n",
                "    except Exception as e:\n",
                "        print(f\"Error making prediction: {e}\")\n",
                "        return None"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run the Demo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "async def run_demo():\n",
                "    print(\"Starting Ray Serve demo: Training and inference happening simultaneously\")\n",
                "\n",
                "    # Start Ray Serve\n",
                "    serve.start()\n",
                "\n",
                "    # Deploy the predictor\n",
                "    predictor = serve.run(ModelPredictor.bind(), route_prefix=\"/predict\")\n",
                "    print(\"Model predictor deployed\")\n",
                "\n",
                "    # Wait for deployment to be ready\n",
                "    while True:\n",
                "        status = serve.status()\n",
                "        if status.applications[\"default\"].deployments[\"predictor\"].status == DeploymentStatus.HEALTHY:\n",
                "            break\n",
                "        print(\"Waiting for deployment to be ready...\")\n",
                "        time.sleep(1)\n",
                "\n",
                "    print(\"Predictor is ready to serve requests\")\n",
                "\n",
                "    # Run inference in a loop\n",
                "    def inference_loop():\n",
                "        print(\"Starting inference loop...\")\n",
                "        for i in range(20):\n",
                "            print(f\"\\nInference request {i + 1}:\")\n",
                "            result = make_prediction()\n",
                "            time.sleep(3)  # Make a request every 3 seconds\n",
                "\n",
                "        print(\"Inference loop complete!\")\n",
                "\n",
                "    # Start inference in a separate thread\n",
                "    inference_thread = threading.Thread(target=inference_loop)\n",
                "    inference_thread.daemon = True\n",
                "    inference_thread.start()\n",
                "\n",
                "    # Start training in the background and update the model as training progresses\n",
                "    model_futures = []\n",
                "    for i in range(5):  # Train 5 iterations of the model\n",
                "        # Start a training task\n",
                "        model_future = train_model.remote(i)\n",
                "        model_futures.append(model_future)\n",
                "\n",
                "        # Wait for this training iteration to complete\n",
                "        model, accuracy = ray.get(model_future)\n",
                "\n",
                "        # Get a handle to the deployment and update the model\n",
                "        handle = serve.get_deployment_handle(\"predictor\", app_name=\"default\")\n",
                "        await handle.update_model.remote(model, accuracy, i + 1)\n",
                "\n",
                "        # Small delay between training iterations\n",
                "        if i < 4:  # Don't sleep after the last iteration\n",
                "            time.sleep(8)\n",
                "\n",
                "    # Wait for the inference thread to complete\n",
                "    inference_thread.join()\n",
                "\n",
                "    print(\"Demo completed!\")\n",
                "\n",
                "    # Shutdown Ray Serve\n",
                "    serve.shutdown()\n",
                "\n",
                "    # Shutdown Ray\n",
                "    ray.shutdown()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Execute the Demo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import asyncio\n",
                "await run_demo()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('ray_serve_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 