## Frontend: Structured Generation Language (SGLang)
The frontend language can be used with local models or API models. It is an alternative to the OpenAI API. You may found it easier to use for complex prompting workflow.

### Quick Start
The example below shows how to use sglang to answer a mulit-turn question.

#### Using Local Models
First, launch a server with
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Then, connect to the server and answer a multi-turn question.

```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

print(state["answer_1"])
```

#### Using OpenAI Models
Set the OpenAI API Key
```
export OPENAI_API_KEY=sk-******
```

Then, answer a multi-turn question.
```python
from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(OpenAI("gpt-3.5-turbo"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

print(state["answer_1"])
```

#### More Examples

Anthropic and VertexAI (Gemini) models are also supported.
You can find more examples at [examples/quick_start](examples/frontend_language/quick_start).

### Language Feature
To begin with, import sglang.
```python
import sglang as sgl
```

`sglang` provides some simple primitives such as `gen`, `select`, `fork`, `image`.
You can implement your prompt flow in a function decorated by `sgl.function`.
You can then invoke the function with `run` or `run_batch`.
The system will manage the state, chat template, parallelism and batching for you.

The complete code for the examples below can be found at [readme_examples.py](examples/frontend_language/usage/readme_examples.py)

#### Control Flow
You can use any Python code within the function body, including control flow, nested function calls, and external libraries.

```python
@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ". "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"]) + ". "

    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "search engine":
        s += "The key word to search is" + sgl.gen("word")
```

#### Parallelism
Use `fork` to launch parallel prompts.
Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.

```python
@sgl.function
def tip_suggestion(s):
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")
```

#### Multi-Modality
Use `sgl.image` to pass an image as input.

```python
@sgl.function
def image_qa(s, image_file, question):
    s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256)
```

See also [srt_example_llava.py](examples/frontend_language/quick_start/local_example_llava_next.py).

#### Constrained Decoding
Use `regex` to specify a regular expression as a decoding constraint.
This is only supported for local models.

```python
@sgl.function
def regular_expression_gen(s):
    s += "Q: What is the IP address of the Google DNS servers?\n"
    s += "A: " + sgl.gen(
        "answer",
        temperature=0,
        regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
    )
```

#### JSON Decoding
Use `regex` to specify a JSON schema with a regular expression.

```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)

@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
```

See also [json_decode.py](examples/frontend_language/usage/json_decode.py) for an additional example of specifying formats with Pydantic models.

#### Batching
Use `run_batch` to run a batch of requests with continuous batching.

```python
@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True
)
```

#### Streaming
Add `stream=True` to enable streaming.

```python
@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

state = text_qa.run(
    question="What is the capital of France?",
    temperature=0.1,
    stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

#### Roles

Use `sgl.system`， `sgl.user` and `sgl.assistant` to set roles when using Chat models. You can also define more complex role prompts using begin and end tokens.

```python
@sgl.function
def chat_example(s):
    s += sgl.system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += sgl.assistant_begin()
    s += "Answer: " + sgl.gen(max_tokens=100, stop="\n")
    s += sgl.assistant_end()
```

#### Tips and Implementation Details
- The `choices` argument in `sgl.gen` is implemented by computing the [token-length normalized log probabilities](https://blog.eleuther.ai/multiple-choice-normalization/) of all choices and selecting the one with the highest probability.
- The `regex` argument in `sgl.gen` is implemented through autoregressive decoding with logit bias masking, according to the constraints set by the regex. It is compatible with `temperature=0` and `temperature != 0`.
