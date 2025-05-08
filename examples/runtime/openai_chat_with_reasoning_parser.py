"""
Usage:
1) Launch the server in one terminal:
   python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --reasoning-parser qwen3

2) Run this script in another terminal:
   python3 openai_chat_with_reasoning_parser.py

This example demonstrates usage of the reasoning parser.
Currently this feature is supported for Deepseek R1 and Qwen3.
"""

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {
            "role": "user",
            "content": "Give me a short introduction to large language models.",
        },
    ],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    extra_body={"top_k": 20, "separate_reasoning": True},
)

print("==== Reasoning ====")
print(chat_response.choices[0].message.reasoning_content)

print("==== Text ====")
print(chat_response.choices[0].message.content)
