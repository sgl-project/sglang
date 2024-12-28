# Tools Calling

## Supported Models

Currently, we added the support for tools calling in the following models:
  - Llama 3.2 models
  - Llama 3.1 models
  - Qwen 2.5 models
  - InternLM Models

## Usage

Checkout an interactive notebook for more examples at `sglang/docs/backend/function_tooling.ipynb`

```python
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
    }
  }
]

messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

client = OpenAI(api_key='YOUR_API_KEY',base_url='http://0.0.0.0:30000/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)

'''

ChatCompletion(id='d6f620e1767e490d85b5ce45c15151cf', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='0', function=Function(arguments='{"a": "3", "b": "5"}', name='add'), type='function')]), matched_stop=128008)], created=1735411703, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=23, prompt_tokens=198, total_tokens=221, completion_tokens_details=None, prompt_tokens_details=None))

'''
```

## How to support a new model?

For adding support of more different models: \n
 1. Update the `TOOLS_TAG_LIST` in `sglang/srt/utils.py` with the tool tag used by the model.
 2. Add support in `parse_tool_response` function for converting into tool calls `sglang/srt/utils.py`
