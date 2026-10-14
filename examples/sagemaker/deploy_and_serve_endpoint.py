import json

import boto3
from sagemaker import serializers
from sagemaker.model import Model
from sagemaker.predictor import Predictor

boto_session = boto3.session.Session()
sm_client = boto_session.client("sagemaker")
sm_role = boto_session.resource("iam").Role("SageMakerRole").arn

endpoint_name = "<YOUR_ENDPOINT_NAME>"
image_uri = "<YOUR_DOCKER_IMAGE_URI>"
model_id = (
    "<YOUR_MODEL_ID>"  # eg: Qwen/Qwen3-0.6B from https://huggingface.co/Qwen/Qwen3-0.6B
)
hf_token = "<YOUR_HUGGINGFACE_TOKEN>"
prompt = "<YOUR_ENDPOINT_PROMPT>"

model = Model(
    name=endpoint_name,
    image_uri=image_uri,
    role=sm_role,
    env={
        "SM_SGLANG_MODEL_PATH": model_id,
        "HF_TOKEN": hf_token,
    },
)
print("Model created successfully")
print("Starting endpoint deployment (this may take 10-15 minutes)...")

endpoint_config = model.deploy(
    instance_type="ml.g5.12xlarge",
    initial_instance_count=1,
    endpoint_name=endpoint_name,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    wait=True,
)
print("Endpoint deployment completed successfully")


print(f"Creating predictor for endpoint: {endpoint_name}")
predictor = Predictor(
    endpoint_name=endpoint_name,
    serializer=serializers.JSONSerializer(),
)

payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2400,
    "temperature": 0.01,
    "top_p": 0.9,
    "top_k": 50,
}
print(f"Sending inference request with prompt: '{prompt[:50]}...'")
response = predictor.predict(payload)
print("Inference request completed successfully")

if isinstance(response, bytes):
    response = response.decode("utf-8")

if isinstance(response, str):
    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        print("Warning: Response is not valid JSON. Returning as string.")

print(f"Received model response: '{response}'")
