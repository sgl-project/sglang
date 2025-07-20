import asyncio
import os

import requests
import runpod
from engine import SGlangEngine
from utils import process_response

# Initialize the engine
engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()


def get_max_concurrency(default=300):
    """
    Returns the maximum concurrency value.
    By default, it uses 50 unless the 'MAX_CONCURRENCY' environment variable is set.

    Args:
        default (int): The default concurrency value if the environment variable is not set.

    Returns:
        int: The maximum concurrency value.
    """
    return int(os.getenv("MAX_CONCURRENCY", default))


async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]

    # Case 1: full OpenAI style payload where caller already specifies the route.
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get(
            "openai_input"
        )

        openai_url = f"{engine.base_url}" + openai_route
        headers = {"Content-Type": "application/json"}

        response = requests.post(openai_url, headers=headers, json=openai_input)
        # Process the streamed response
        if openai_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    yield decoded_chunk

    # Case 2: payload looks like OpenAI chat/completions but omits the wrapper.
    elif "messages" in job_input:
        openai_url = f"{engine.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Make sure model is set; fall back to default.
        if "model" not in job_input:
            job_input["model"] = engine.model or "default"

        response = requests.post(openai_url, headers=headers, json=job_input)

        if job_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    # Case 3: assume user meant the native /generate endpoint.
    else:
        generate_url = f"{engine.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        # Directly pass `job_input` to `json`. Can we tell users the possible fields of `job_input`?
        response = requests.post(generate_url, json=job_input, headers=headers)

        if response.status_code == 200:
            yield response.json()
        else:
            yield {
                "error": f"Generate request failed with status code {response.status_code}",
                "details": response.text,
            }


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": get_max_concurrency,
        "return_aggregate_stream": True,
    }
)
