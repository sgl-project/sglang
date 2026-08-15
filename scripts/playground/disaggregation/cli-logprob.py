prompt = "The capital of france is "

import json

import requests

response = requests.post(
    "http://0.0.0.0:8000/generate",
    json={
        "text": prompt,
        "sampling_params": {"temperature": 0},
        "return_logprob": True,
        "return_input_logprob": True,
        "logprob_start_len": 0,
    },
)

j = response.json()
input_logprobs = j["meta_info"]["input_token_logprobs"]
output_logprobs = j["meta_info"]["output_token_logprobs"]

print(len(input_logprobs), len(output_logprobs))
