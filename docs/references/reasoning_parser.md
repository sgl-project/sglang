# Reasoning Parser

SGLang supports parsing reasoning content our from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

The contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.

## Supported Models

Currently, SGLang supports the following reasoning models:
- [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `<think>` and `</think>` tags.

## Usage

There are two ways to enable reasoning parsing:

1) Enable the reasoning parser when starting the SGLang Server by setting the `--enable-reasoning` and `--reasoning-parser` options. The `--reasoning-parser` option specifies the reasoning parser to extract the reasoning content and final answer.

```bash
python -m sglang.launch_server --host 0.0.0.0 \
--model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
--enable-reasoning --reasoning-parser deepseek-r1
```

2) Specify on a per-request basis by setting the `separate_reasoning` body field on a `/chat/completions` request.

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"messages":[{"role":"user","content":"Compute 1+3"}],"max_tokens":100,"model":"deepseek-r1","stream":true,"separate_reasoning":true}' http://0.0.0.0:30000/v1/chat/completions
```

There is another body param which can be set to buffer the reasoning traces to be sent in one chunk after the closing `</think>` tag, `"stream_reasoning": false`.

### Non-streaming Request

Make a request to the reasoning model, get the reasoning content and final answer.

Using OpenAI python api:
```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="deepseek-r1:14b",
    messages=[{"role": "user", "content": "Compute 1+3"}],
    max_tokens=1024,
    stream=False
)

response.choices[0].message.reasoning_content
# 'First, I recognize that the problem requires adding the numbers 1 and 3.\n\nNext, I identify the numbers to be added, which are 1 and 3.\n\nThen, I perform the addition operation: 1 plus 3 equals 4.\n\nFinally, I conclude that the sum of 1 and 3 is 4.\n'
response.choices[0].message.content
# \n\nTo compute \\(1 + 3\\), follow these simple steps:\n\n1. **Identify the numbers to add:**  \n   The numbers are **1** and **3**.\n\n2. **Add the numbers together:**  \n   \\[\n   1 + 3 = 4\n   \\]\n\n3. **Write the final answer:**  \n   The sum of \\(1 + 3\\) is \\(\\boxed{4}\\).'
```

### Streaming Request

`reasoning_content` is available in the `delta` field of the streaming response.

Using OpenAI python api:

```python
# ... Initialize the client as before ...

response = client.chat.completions.create(
    model="deepseek-r1:14b",
    messages=[{"role": "user", "content": "Compute 1+3"}],
    max_tokens=1024,
    stream=True
)
reasoning_content = ""
content = ""
for chunk in response:
    if chunk.choices[0].delta.content:
      content += chunk.choices[0].delta.content
    elif chunk.choices[0].delta.reasoning_content:
      reasoning_content += chunk.choices[0].delta.reasoning_content

reasoning_content
# 'I need to calculate the sum of 1 and 3. \n\nFirst, I identify the numbers involved in the addition: 1 and 3.\n\nNext, I add these two numbers together to find the total.\n\nFinally, the result of the addition is 4.\n'
content
# '\n\n**Solution:**\n\nWe need to compute the sum of 1 and 3.\n\n1. **Identify the numbers to add:**\n   - Number 1\n   - Number 3\n\n2. **Add the numbers together:**\n   \\[\n   1 + 3 = 4\n   \\]\n\n3. **Final Answer:**\n   \\[\n   \\boxed{4}\n   \\]'
```


## Supporting New Reasoning Models

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningParser` in `python/sglang/srt/reasoning_parser.py`.

```python
class BaseReasoningParser:
    """Base class for reasoning parser."""

    def __init__(self):
        self._buffer = ""

    def detect_and_parse(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect and parse the text, return reasoning_content and content."""
        raise NotImplementedError

    def parse_streaming_increment(
        self, new_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the new text incrementally, return reasoning_content and content."""
        raise NotImplementedError
```

And specify the reasoning parser for new reasoning models accordingly.

```python
class ReasoningParser:
    """Reasoning parser for different reasoning models."""

    # Specify the reasoning parser for each reasoning model here
    ReasoningParserDict: Dict[str, Type[BaseReasoningParser]] = {
        "deepseek-r1": DeepSeekR1ReasoningParser
    }

    def __init__(self, reasoning_parser: str):
        self.parser = self.ReasoningParserDict[reasoning_parser]()

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Non-streaming parsing for reasoning models.
        Return: reasoning_content, content
        """
        return self.parser.detect_and_parse(full_text)

    def parse_stream_chunk(self, chunk_text: str):
        """
        Streaming parsing for reasoning models.
        Return: reasoning_content, content
        """
        return self.parser.parse_streaming_increment(chunk_text)
```
