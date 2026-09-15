"""
Usage:
python -m sglang.launch_server --model meta-llama/Llama-2-7b-hf --port 30000
python vertex_predict.py

This example shows the request and response formats of the prediction route for
Google Cloud Vertex AI Online Predictions.

Vertex AI SDK for Python is recommended for deploying models to Vertex AI
instead of a local server. After deploying the model to a Vertex AI Online
Prediction Endpoint, send requests via the Python SDK:

response = endpoint.predict(
    instances=[
        {"text": "The capital of France is"},
        {"text": "What is a car?"},
    ],
    parameters={"sampling_params": {"max_new_tokens": 16}},
)
print(response.predictions)

More details about get online predictions from Vertex AI can be found at
https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions.
"""

from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class VertexPrediction:
    predictions: List


class LocalVertexEndpoint:
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:30000"

    def predict(self, instances: List[dict], parameters: Optional[dict] = None):
        response = requests.post(
            self.base_url + "/vertex_generate",
            json={
                "instances": instances,
                "parameters": parameters,
            },
        )
        return VertexPrediction(predictions=response.json()["predictions"])


endpoint = LocalVertexEndpoint()

# Predict with a single prompt.
response = endpoint.predict(instances=[{"text": "The capital of France is"}])
print(response.predictions)

# Predict with multiple prompts and parameters.
response = endpoint.predict(
    instances=[
        {"text": "The capital of France is"},
        {"text": "What is a car?"},
    ],
    parameters={"sampling_params": {"max_new_tokens": 16}},
)
print(response.predictions)
