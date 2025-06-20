"""
A test script to verify that the Qwen2.5-VL model is working correctly with pipeline parallelism in SGLang.

This script sends a request with an image and a text prompt to a running SGLang server
and prints the model's response.

=====================================================================================
==> Step 1: Launch the SGLang Server with the launch_server.sh script <==
=====================================================================================
Open a terminal and follow these steps:

1. Modify the `launch_server.sh` script:
   Ensure the `MODEL_PATH` variable in the script points to the correct directory of your model.

2. Grant execute permissions to the script:
   chmod +x launch_server.sh

3. Run the script to start the server:
   ./launch_server.sh

=====================================================================================
==> Step 2: Run this Test Script <==
=====================================================================================
Open another terminal and run this script:

python3 sglang/python/sglang/test/test_pp_qwen_vl.py

The script will then connect to the server, send the request, and print the model's output.
"""

import base64
from openai import OpenAI

# 1. Configuration
# =================================================================
# SGLang server URL
BASE_URL = "http://127.0.0.1:26000/v1"
API_KEY = "sglang"

# !!!IMPORTANT!!! This path must exactly match the path configured in your launch_server.sh
MODEL_PATH = "/path/to/your/Qwen2.5-VL-32B-Instruct/"

# !!!IMPORTANT!!! Please replace this with the path to your local image file for testing
IMAGE_PATH = "/path/to/your/image.png"

# The prompt you want to ask about the image
USER_PROMPT = "Please describe this image in detail."


# 2. Helper Function: Encode Image to Base64
# =================================================================
def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Please ensure the file exists and update the IMAGE_PATH variable.")
        exit(1)
    except Exception as e:
        print(f"Error reading or encoding the image: {e}")
        exit(1)


# 3. Main Test Function
# =================================================================
def run_test():
    """Connects to the SGLang server and runs the test."""
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    print(f"--- Test Details ---")
    print(f"Server URL: {BASE_URL}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Image Path: {IMAGE_PATH}")
    print("--------------------")

    print(f"\n[1/3] Loading and encoding image: {IMAGE_PATH}...")
    base64_image = encode_image_to_base64(IMAGE_PATH)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    print("Image encoded successfully.")

    print("\n[2/3] Sending request to the model...")
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        print("Request successful.")

        print("\n[3/3] Model Response:")
        print("="*50)
        print(response.choices[0].message.content)
        print("="*50)

    except Exception as e:
        print(f"\nRequest failed with error: {e}")
        print("Please check the following:")
        print(f"1. Is the SGLang server running at {BASE_URL}?")
        print("2. Did you successfully start the server with ./launch_server.sh?")
        print("3. Is the network connection stable?")


if __name__ == "__main__":
    run_test() 