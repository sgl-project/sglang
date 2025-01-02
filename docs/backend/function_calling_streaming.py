from openai import OpenAI
import json

# Define tools
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

# Messages from the user
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston today? Please respond with the format: Today's weather is :{function call result}",
    }
]

# Initialize OpenAI-like client
client = OpenAI(api_key="YOUR_API_KEY", base_url="http://0.0.0.0:30520/v1")

# Use the first available model from the backend
model_name = client.models.list().data[0].id

# Non-streaming mode test
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

# Streaming mode test
print("Streaming response:")
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=True,  # Enable streaming
    tools=tools,
)

# Handle streaming responses, combine different chunks
chunks = []
for chunk in response_stream:
    chunks.append(chunk)
    print(chunk)  # Optionally print each chunk to observe its content

# Parse and combine function call arguments
arguments = []
for chunk in chunks:
    choice = chunk.choices[0]
    delta = choice.delta
    if delta.tool_calls:
        tool_call = delta.tool_calls[0]
        if tool_call.function.name:
            print(f"Streamed function call name: {tool_call.function.name}")

        if tool_call.function.arguments:
            arguments.append(tool_call.function.arguments)
            print(f"Streamed function call arguments: {tool_call.function.arguments}")

# Combine all argument fragments
full_arguments = "".join(arguments)
print(f"Final streamed function call arguments: {full_arguments}")

# Add user message and function call result to the message list
messages.append(
    {
        "role": "user",
        "content": "",
        "tool_calls": {"name": "get_current_weather", "arguments": full_arguments},
    }
)


# Define the actual function for getting current weather
def get_current_weather(location: str, unit: str):
    # Here you can integrate an actual weather API
    return f"The weather in {location} is 85 degrees {unit}. It is partly cloudy, with highs in the 90's."


# Simulate tool call
available_tools = {"get_current_weather": get_current_weather}

# Parse JSON arguments
try:
    call_data = json.loads(full_arguments)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    call_data = {}

# Call the corresponding tool function
if "name" in messages[-1]["tool_calls"]:
    tool_name = messages[-1]["tool_calls"]["name"]
    if tool_name in available_tools:
        tool_to_call = available_tools[tool_name]
        result = tool_to_call(**call_data)
        print(f"Function call result: {result}")
        messages.append({"role": "tool", "content": result, "name": tool_name})
    else:
        print(f"Unknown tool name: {tool_name}")
else:
    print("Function call name not found.")

print(f"Messages: {messages}")

# Use the function call result to continue the conversation
chat_completion_final = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools,
)

print("\nFinal Chat Completion:")
print(chat_completion_final)
