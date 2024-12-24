# Tools Calling

Currently, we added the support for tools calling in the following models:
  - Llama 3.2 models
  - Llama 3.1 models
  - Qwen 2.5 models
  - InternLM Models

For adding support of more different architectures: \n
 1. Add Tool tag present in model in line 1055 `sglang/srt/openai_api/adapter.py`
 2. Add support in parsing function for converting into tool calls `sglang/srt/utils.py`


## Single Round Invocation
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
{'id': 'dc98cb31f2674f34bd85513ad3eadfb2', 'object': 'chat.completion', 'created': 1734879695, 'model': 'meta-llama/Llama-3.2-1B-Instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<|python_tag|>{"name": "get_current_weather", "parameters": {"location": "Boston", "unit": "celsius"}}', 'tool_calls': [{'id': '0', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': '{"location": "Boston", "unit": "celsius"}'}}]}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 128008}], 'usage': {'prompt_tokens': 220, 'total_tokens': 246, 'completion_tokens': 26, 'prompt_tokens_details': None}}
'''

```

## Multi-Round Invocation

```python
from openai import OpenAI


def add(a: int, b: int):
    return a + b


def mul(a: int, b: int):
    return a * b


tools = [{
    'type': 'function',
    'function': {
        'name': 'add',
        'description': 'Compute the sum of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}, {
    'type': 'function',
    'function': {
        'name': 'mul',
        'description': 'Calculate the product of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}]
messages = [{'role': 'user', 'content': 'Compute (3+5)*2'}]

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func1_name = response.choices[0].message.tool_calls[0].function.name
func1_args = response.choices[0].message.tool_calls[0].function.arguments
func1_out = eval(f'{func1_name}(**{func1_args})')
print(func1_out)

messages.append(response.choices[0].message)
messages.append({
    'role': 'tool',
    'content': f'3+5={func1_out}',
    'tool_call_id': response.choices[0].message.tool_calls[0].id
})
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func2_name = response.choices[0].message.tool_calls[0].function.name
func2_args = response.choices[0].message.tool_calls[0].function.arguments
func2_out = eval(f'{func2_name}(**{func2_args})')
print(func2_out)

'''
ChatCompletion(id='40e0a1463bdc4d5393c89137702c56e4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='<|python_tag|>{"name": "add", "parameters": {"a": "3", "b": "5"}}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='0', function=Function(arguments='{"a": "3", "b": "5"}', name='add'), type='function')]), matched_stop=128008)], created=1734879706, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=23, prompt_tokens=300, total_tokens=323, completion_tokens_details=None, prompt_tokens_details=None))
35
ChatCompletion(id='800dd951b8594ee78f32d0390d4dbf0e', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='<|python_tag|>{"name": "add", "parameters": {"a": "35", "b": "2"}}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='0', function=Function(arguments='{"a": "35", "b": "2"}', name='add'), type='function')]), matched_stop=128008)], created=1734879706, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=23, prompt_tokens=340, total_tokens=363, completion_tokens_details=None, prompt_tokens_details=None))
352
'''
```
