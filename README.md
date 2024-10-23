<div align="center"  id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)

</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-07-25-sglang-llama3/) | [**Paper**](https://arxiv.org/abs/2312.07104) | [**Slides**](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_dev_day_v2.pdf) | [**Learn More**](https://github.com/sgl-project/sgl-learning-materials) | [**Join Slack**](https://join.slack.com/t/sgl-fru7574/shared_invite/zt-2ngly9muu-t37XiH87qvD~6rVBTkTEHw) |
[**Join Bi-Weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?usp=sharing) |

## News
- [2024/10] 🔥 The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/04] SGLang is used by the official **LLaVA-NeXT (video)** release ([blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, continuous batching, token attention (paged attention), tensor parallelism, FlashInfer kernels, chunked prefill, and quantization (INT4/FP8/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.) and embedding models (e5-mistral), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

## Contents
- [Install](#install)
- [Backend: SGLang Runtime (SRT)](#backend-sglang-runtime-srt)
- [Frontend: Structured Generation Language (SGLang)](#frontend-structured-generation-language-sglang)
- [Benchmark And Performance](#benchmark-and-performance)
- [Roadmap](#roadmap)
- [Citation And Acknowledgment](#citation-and-acknowledgment)

## Install

You can install SGLang using any of the methods below.

### Method 1: With pip
```
pip install --upgrade pip
pip install "sglang[all]"

# Install FlashInfer accelerated kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Important: Please check the [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html) to install the proper version according to your PyTorch and CUDA versions.**

### Method 2: From source
```
# Use the last release branch
git clone -b v0.3.4.post1 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"

# Install FlashInfer accelerated kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Important: Please check the [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html) to install the proper version according to your PyTorch and CUDA versions.**

### Method 3: Using docker
The docker images are available on Docker Hub as [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).
Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

### Method 4: Using docker compose

<details>
<summary>More</summary>

> This method is recommended if you plan to serve it as a service.
> A better approach is to use the [k8s-sglang-service.yaml](docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.
</details>

### Method 5: Run on Kubernetes or Clouds with SkyPilot

<details>
<summary>More</summary>

To deploy on Kubernetes or 12+ clouds, you can use [SkyPilot](https://github.com/skypilot-org/skypilot).

1. Install SkyPilot and set up Kubernetes cluster or cloud access: see [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
2. Deploy on your own infra with a single command and get the HTTP API endpoint:
<details>
<summary>SkyPilot YAML: <code>sglang.yaml</code></summary>

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```
</details>

```bash
# Deploy on any cloud or Kubernetes cluster. Use --cloud <cloud> to select a specific cloud provider.
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang
```
3. To further scale up your deployment with autoscaling and failure recovery, check out the [SkyServe + SGLang guide](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve).
</details>


### Common Notes
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
- If you only need to use the OpenAI backend, you can avoid installing other dependencies by using `pip install "sglang[openai]"`.

## Backend: SGLang Runtime (SRT)
The SGLang Runtime (SRT) is an efficient serving engine.

### Quick Start
Launch a server
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Send a request
```
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

Learn more about the argument specification, streaming, and multi-modal support [here](docs/en/sampling_params.md).

### OpenAI Compatible API
In addition, the server supports OpenAI-compatible APIs.

```python
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
	model="default",
	prompt="The capital of France is",
	temperature=0,
	max_tokens=32,
)
print(response)

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)

# Text embedding
response = client.embeddings.create(
    model="default",
    input="How are you today",
)
print(response)
```

It supports streaming, vision, and almost all features of the Chat/Completions/Models/Batch endpoints specified by the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/).

### Additional Server Arguments
- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
```
- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
```
- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
```
- See [hyperparameter_tuning.md](docs/en/hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
```
- To enable the experimental overlapped scheduler, add `--enable-overlap-scheduler`. It overlaps CPU scheduler with GPU computation and can accelerate almost all workloads. This does not work for constrained decoding currenly.
- To enable torch.compile acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. This does not work for FP8 currenly.
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports various quantization strategies.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](docs/en/custom_chat_template.md).
- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 1
```
 
### Engine Without HTTP Server

We also provide an inference engine **without a HTTP server**. For example,

```python
import sglang as sgl

def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95}
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == "__main__":
    main()
```

This can be used for offline batch inference and building custom servers.
You can view the full example [here](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine).

### Supported Models

**Generative Models**
- Llama / Llama 2 / Llama 3 / Llama 3.1
- Mistral / Mixtral / Mistral NeMo
- Gemma / Gemma 2
- Qwen / Qwen 2 / Qwen 2 MoE
- DeepSeek / DeepSeek 2
- OLMoE
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --port=30000 --chat-template=chatml-llava`
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-72b-ov --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](test/srt/test_vision_openai_server.py)
- LLaVA 1.5 / 1.6 / NeXT
  - `python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000 --tp-size=1 --chat-template=llava_llama_3`
  - `python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](test/srt/test_vision_openai_server.py)
- Yi-VL
- StableLM
- Command-R
- DBRX
- Grok
- ChatGLM
- InternLM 2
- Exaone 3
- BaiChuan2
- MiniCPM / MiniCPM 3
- XVERSE / XVERSE MoE
- SmolLM
- GLM-4

**Embedding Models**

- e5-mistral
- gte-Qwen2
  - `python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding`

Instructions for supporting a new model are [here](docs/en/model_support.md).

#### Use Models From ModelScope
<details>
<summary>More</summary>

To use a model from [ModelScope](https://www.modelscope.cn), set the environment variable SGLANG_USE_MODELSCOPE.
```
export SGLANG_USE_MODELSCOPE=true
```
Launch [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) Server
```
SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

Or start it by docker.
```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```
  
</details>

#### Run Llama 3.1 405B
<details>
<summary>More</summary>

```bash
# Run 405B (fp8) on a single node
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8

# Run 405B (fp16) on two nodes
## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0 --disable-cuda-graph

## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 1 --disable-cuda-graph
```

</details>

### Benchmark Performance

- Benchmark a single static batch by running the following command without launching a server. The arguments are the same as for `launch_server.py`.
  Note that this is not a dynamic batching server, so it may run out of memory for a batch size that a real server can handle.
  A real server truncates the prefill into several batches, while this unit test does not. For accurate large batch testing, please use `sglang.bench_serving` instead.
  ```
  python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 32 --input-len 256 --output-len 32
  ```
- Benchmark online serving. Launch a server first and run the following command.
  ```
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```

## Frontend: Structured Generation Language (SGLang)
The frontend language can be used with local models or API models. It is an alternative to the OpenAI API. You may found it easier to use for complex prompting workflow.

### Quick Start
The example below shows how to use sglang to answer a multi-turn question.

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

## Benchmark And Performance
Learn more in our release blogs: [v0.2](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3](https://lmsys.org/blog/2024-09-04-sglang-v0-3/).

## Roadmap
[Development Roadmap (2024 Q4)](https://github.com/sgl-project/sglang/issues/1487)

## Citation And Acknowledgment
Please cite our paper, [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), if you find the project useful.
We also learned from the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).


<p align="center">
  <a href="#sglangtop" target="_blank">
  <bold>Back To Top </bold>
  </a>
</p>
