To refine and clean up the README you've provided for the SGLang project, I'll focus on improving clarity, organization, and conciseness. This includes providing clear installation instructions, simplifying steps where possible, and ensuring the document is easy to navigate. Here's a revised version:

---

<div align="center">
    <img src="assets/logo.png" alt="SGLang Logo" width="400">
</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-01-17-sglang/) | [**Paper**](https://arxiv.org/abs/2312.07104) |

**SGLang** is a structured generation language tailored for large language models (LLMs), enhancing interaction speed and control by integrating an adaptable frontend language with a high-performance runtime system.

### Key Features
- **Flexible Front-End Language**: Facilitates programming of LLM applications, supporting chained generations, advanced prompts, control flows, multi-modalities, parallelism, and external interactions.
- **High-Performance Runtime with RadixAttention**: Boosts complex LLM program execution through KV cache reuse, continuous batching, and tensor parallelism.

## Getting Started

### Installation

#### SGLang Setup
```sh
cd PATH_TO/sglang_video
pip install --upgrade pip
pip install -e "python[all]"
```


#### Dependency Installation

Install the necessary Python packages:

```sh
pip3 install huggingface_hub hf_transfer outlines==0.0.34 opencv-python-headless transformers==4.39.2
```


#### Additional Notes

- Make sure to replace `PATH_TO` with the actual path to your `sglang_video` directory.
- Adjust the `repo_id` and `local_dir` as needed based on the specific models you intend to use.


### System Requirements
- CUDA >= 12.1. Specific GPU models may require different versions of the Triton compiler:
  - NVIDIA T4: `pip install "triton>=2.2.0"`
  - NVIDIA V100: Install the [nightly version of Triton](https://triton-lang.org/main/getting-started/installation.html).
- For OpenAI backend usage only: `pip install "sglang[openai]"`

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

### Streaming
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

### Tips and Implementation Details
- The `choices` argument in `sgl.gen` is implemented by computing the normalized log probabilities of all choices and selecting the one with the highest probability.
- The `regex` argument in `sgl.gen` is implemented through autoregressive decoding with logit bias masking, according to the constraints set by the regex.

## Backend: SGLang Runtime (SRT)
The SGLang Runtime (SRT) is designed to work best with the SGLang frontend.
However, it can also be used as a standalone API server.
In this case, the [RadixAttention](https://arxiv.org/abs/2312.07104) can still greatly accelerate many use cases with automatic KV cache reuse.

### Usage
Launch a server
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
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
Learn more about the argument format [here](docs/sampling_params.md).

### OpenAI Compatible API

In addition, the server supports an experimental OpenAI-compatible API.

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
```

In above example, the server uses the chat template specified in the model tokenizer.
You can override the chat template if needed when launching the server:

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template llama-2
```

If the chat template you are looking for is missing, you are welcome to contribute it.
Meanwhile, you can also temporary register your chat template as follows:

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

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template ./my_model_template.json
```

### Additional Arguments
- Add `--tp 2` to enable tensor parallelism.
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --tp 2
```
- If you see out-of-memory errors during serving, please try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --mem-fraction-static 0.7
```
- You can turn on [flashinfer](docs/flashinfer.md) to accelerate the inference by using highly optimized CUDA kernels.

### Supported Models
- Llama
- Mistral
- Mixtral
- Qwen / Qwen 2
- Gemma
  - Please add a new flag `--attention-reduce-in-fp32` to avoid some precision errors.
  - `python -m sglang.launch_server --model-path google/gemma-7b-it --port 30000 --attention-reduce-in-fp32`
- LLaVA
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000`
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000`
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 3000`
- Yi-VL
  - see [srt_example_yi_vl.py](examples/quick_start/srt_example_yi_vl.py).
- StableLM
- Command-R
- DBRX
- AWQ/GPTQ/Marlin quantization

Instructions for supporting a new model are [here](https://github.com/sgl-project/sglang/blob/main/docs/model_support.md).

## Benchmark And Performance

- Llama-7B on NVIDIA A10G, FP16, Tensor Parallelism=1
![llama_7b](assets/llama_7b.jpg)

- Mixtral-8x7B on NVIDIA A10G, FP16, Tensor Parallelism=8
![mixtral_8x7b](assets/mixtral_8x7b.jpg)

Learn more [here](docs/benchmark_results.md).

## Roadmap
https://github.com/sgl-project/sglang/issues/157

## Citation And Acknowledgment
```
@misc{zheng2023efficiently,
      title={Efficiently Programming Large Language Models using SGLang},
      author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
      year={2023},
      eprint={2312.07104},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

1. **Prepare Environment**:
   ```sh
   cd PATH_TO/sglang_video
   ```

2. **Launch and Run on (K) Nodes**:
   - First node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K 0 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
     ```
   - Second node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K 1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
     ```
   - The K node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K K-1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
     ```

Replace `K`, `YOUR_VIDEO_PATH`, `YOUR_MODEL_PATH`, and `FRAMES_PER_VIDEO` with your specific details.