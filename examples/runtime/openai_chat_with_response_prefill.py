"""
Usage:
1) Launch the server in one terminal:
   python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

2) Run this script in another terminal:
   python openai_chat_with_response_prefill.py

This example demonstrates two chat completion calls:
- One with continue_final_message enabled (the final assistant message is used as a prefill).
- One without continue_final_message (the final assistant message remains, starting a new turn).
"""

import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {
        "role": "user",
        "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99.
At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or app—
no matter where you place it in your home.
This affordable little hub brings convenient hands-free control to your smart devices.
</description>
""",
    },
    {"role": "assistant", "content": "{\n"},
]

# Calling the API with continue_final_message enabled.
print("=== Prefill with continue_final_messagem ===")
response_with = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0,
    extra_body={"continue_final_message": True},
)
print(response_with.choices[0].message.content)

# Calling the API without continue_final_message (using default behavior).
print("\n=== Prefill without continue_final_message ===")
response_without = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0,
)
print(response_without.choices[0].message.content)
