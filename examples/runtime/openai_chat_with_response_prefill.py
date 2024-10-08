"""
Usage:
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
python openai_chat.py
"""

import openai
from openai import OpenAI

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {
            "role": "user",
            "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99. At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or app—no matter where you place it in your home. This affordable little hub brings convenient hands-free control to your smart devices.
</description>
""",
        },
        {
            "role": "assistant",
            "content": "{\n",
        },
    ],
    temperature=0,
)

print(response.choices[0].message.content)
