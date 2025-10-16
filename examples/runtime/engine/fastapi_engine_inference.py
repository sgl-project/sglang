"""
FastAPI server example for text generation using SGLang Engine and demonstrating client usage.

Starts the server, sends requests to it, and prints responses.

Usage:
python fastapi_engine_inference.py --model-path Qwen/Qwen2.5-0.5B-Instruct --tp_size 1 --host 127.0.0.1 --port 8000 [--startup-timeout 60]
"""

import os
import subprocess
import time
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Request

import sglang as sgl
from sglang.utils import terminate_process

engine = None


# Use FastAPI's lifespan manager to initialize/shutdown the engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages SGLang engine initialization during server startup."""
    global engine
    # Initialize the SGLang engine when the server starts
    # Adjust model_path and other engine arguments as needed
    print("Loading SGLang engine...")
    engine = sgl.Engine(
        model_path=os.getenv("MODEL_PATH"), tp_size=int(os.getenv("TP_SIZE"))
    )
    print("SGLang engine loaded.")
    yield
    # Clean up engine resources when the server stops (optional, depends on engine needs)
    print("Shutting down SGLang engine...")
    # engine.shutdown() # Or other cleanup if available/necessary
    print("SGLang engine shutdown.")


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_text(request: Request):
    """FastAPI endpoint to handle text generation requests."""
    global engine
    if not engine:
        return {"error": "Engine not initialized"}, 503

    try:
        data = await request.json()
        prompt = data.get("prompt")
        max_new_tokens = data.get("max_new_tokens", 128)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return {"error": "Prompt is required"}, 400

        # Use async_generate for non-blocking generation
        state = await engine.async_generate(
            prompt,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            # Add other parameters like stop, top_p etc. as needed
        )

        return {"generated_text": state["text"]}
    except Exception as e:
        return {"error": str(e)}, 500


# Helper function to start the server
def start_server(args, timeout=60):
    """Starts the Uvicorn server as a subprocess and waits for it to be ready."""
    base_url = f"http://{args.host}:{args.port}"
    command = [
        "python",
        "-m",
        "uvicorn",
        "fastapi_engine_inference:app",
        f"--host={args.host}",
        f"--port={args.port}",
    ]

    process = subprocess.Popen(command, stdout=None, stderr=None)

    start_time = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
            try:
                # Check the /docs endpoint which FastAPI provides by default
                response = session.get(
                    f"{base_url}/docs", timeout=5
                )  # Add a request timeout
                if response.status_code == 200:
                    print(f"Server {base_url} is ready (responded on /docs)")
                    return process
            except requests.ConnectionError:
                # Specific exception for connection refused/DNS error etc.
                pass
            except requests.Timeout:
                # Specific exception for request timeout
                print(f"Health check to {base_url}/docs timed out, retrying...")
                pass
            except requests.RequestException as e:
                # Catch other request exceptions
                print(f"Health check request error: {e}, retrying...")
                pass
            # Use a shorter sleep interval for faster startup detection
            time.sleep(1)

    # If loop finishes, raise the timeout error
    # Attempt to terminate the failed process before raising
    if process:
        print(
            "Server failed to start within timeout, attempting to terminate process..."
        )
        terminate_process(process)  # Use the imported terminate_process
    raise TimeoutError(
        f"Server failed to start at {base_url} within the timeout period."
    )


def send_requests(server_url, prompts, max_new_tokens, temperature):
    """Sends generation requests to the running server for a list of prompts."""
    # Iterate through prompts and send requests
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Sending prompt: '{prompt}'")
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(f"{server_url}/generate", json=payload, timeout=60)

            result = response.json()

            print(f"Prompt: {prompt}\nResponse: {result['generated_text']}")

        except requests.exceptions.Timeout:
            print(f"  Error: Request timed out for prompt '{prompt}'")
        except requests.exceptions.RequestException as e:
            print(f"  Error sending request for prompt '{prompt}': {e}")


if __name__ == "__main__":
    """Main entry point for the script."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=60,
        help="Time in seconds to wait for the server to be ready (default: %(default)s)",
    )
    args = parser.parse_args()

    # Pass the model to the child uvicorn process via an env var
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["TP_SIZE"] = str(args.tp_size)

    # Start the server
    process = start_server(args, timeout=args.startup_timeout)

    # Define the prompts and sampling parameters
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_new_tokens = 64
    temperature = 0.1

    # Define server url
    server_url = f"http://{args.host}:{args.port}"

    # Send requests to the server
    send_requests(server_url, prompts, max_new_tokens, temperature)

    # Terminate the server process
    terminate_process(process)
