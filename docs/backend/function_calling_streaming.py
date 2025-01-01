from openai import OpenAI

# The tools you have defined
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

"""

"""

messages = [
    {"role": "user", "content": "What's the weather like in Boston today? Please respond with the format: Today's weather is :{function call result}"}
]

# Initialize OpenAI-like client
client = OpenAI(api_key="YOUR_API_KEY", base_url="http://0.0.0.0:30000/v1")

# Assume there are some models available in your backend, use the first one for demonstration
model_name = client.models.list().data[0].id

# ---- 1) Non-streaming mode test ----
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,  # Non-streaming
    tools=tools,
)
print("Non-stream response:")
print(response_non_stream)

# ---- 2) Streaming mode test ----
print("Streaming response:")
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=True,   # Enable streaming
    tools=tools,
)

# Note: In streaming mode, the response itself may be a generator/iterator
for chunk in response_stream:
    print(chunk)
