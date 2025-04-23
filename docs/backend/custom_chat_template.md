# Custom Chat Template

**NOTE**: There are two chat template systems in SGLang project. This document is about setting a custom chat template for the OpenAI-compatible API server (defined at [conversation.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/conversation.py)). It is NOT related to the chat template used in the SGLang language frontend (defined at [chat_template.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py)).

By default, the server uses the chat template specified in the model tokenizer from Hugging Face.
It should just work for most official models such as Llama-2/Llama-3.

If needed, you can also override the chat template when launching the server:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template llama-2
```

If the chat template you are looking for is missing, you are welcome to contribute it or load it from a file.

## JSON Format

You can load the JSON format, which is defined by `conversation.py`.

```json
{
  "name": "my_model",
  "system": "<|im_start|>system",
  "user": "<|im_start|>user",
  "assistant": "<|im_start|>assistant",
  "sep_style": "CHATML",
  "sep": "<|im_end|>",
  "stop_str": ["<|im_end|>", "<|im_start|>"]
}
```

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template ./my_model_template.json
```

## Jinja Format

You can also use the [Jinja template format](https://huggingface.co/docs/transformers/main/en/chat_templating) as defined by Hugging Face Transformers.

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template ./my_model_template.jinja
```

### Passing Arguments to Jinja Templates

When using a Jinja-based chat template (either the default template embedded in the tokenizer or a custom template loaded via `--chat-template path/to/template.jinja`), you can pass additional keyword arguments to the template renderer context.

This is done using the `chat_template_kwargs` parameter in the `/v1/chat/completions` request body. This parameter accepts a JSON object (dictionary) where keys are the argument names and values are the corresponding values you want to make available within your Jinja template.

**Example Request:**

```json
{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "chat_template_kwargs": {
    "my_custom_arg": "some_value",
    "another_arg": 123
  }
}
```

**Example Jinja Template (`my_template.jinja`):**

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + eos_token }}
    {% endif %}
{% endfor %}

{# Accessing custom arguments #}
Custom Arg: {{ my_custom_arg }}
Another Arg: {{ another_arg }}
```

**Important Notes:**

*   The `chat_template_kwargs` parameter in the request **only** works with Jinja-based templates. It has no effect when using legacy JSON-based templates (loaded via `--chat-template template_name` or `--chat-template path/to/template.json`).
*   You can also set *global* default arguments using the `--chat-template-kwargs` server launch flag, which accepts a JSON string (e.g., `--chat-template-kwargs '{"global_arg": true}'`).
*   If `chat_template_kwargs` is provided in a specific request, it **completely overrides** any global arguments set via the server flag for that request. If `chat_template_kwargs` is *not* provided in the request, the global arguments (if set) will be used.
