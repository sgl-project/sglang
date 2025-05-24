# DeepSeek Usage

SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended) from Day 0.

This document outlines current optimizations for DeepSeek.
Additionally, the SGLang team is actively developing enhancements following this [Roadmap](https://github.com/sgl-project/sglang/issues/2591).

## Launch DeepSeek V3 with SGLang

To run DeepSeek V3/R1 models, the requirements are as follows:

| Weight Type | Configuration |
|------------|-------------------|
| **Full precision FP8**<br>*(recommended)* | 8 x H200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| **Full precision BF16** | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **Quantized weights (AWQ)** | 8 x H100/800/20 |
| | 8 x A100/A800 |
| **Quantized weights (int8)** | 16 x A100/800 |
| | 32 x L40S |

<style>
.md-typeset__table {
  width: 100%;
}

.md-typeset__table table {
  border-collapse: collapse;
  margin: 1em 0;
  border: 2px solid var(--md-typeset-table-color);
  table-layout: fixed;
}

.md-typeset__table th {
  border: 1px solid var(--md-typeset-table-color);
  border-bottom: 2px solid var(--md-typeset-table-color);
  background-color: var(--md-default-bg-color--lighter);
  padding: 12px;
}

.md-typeset__table td {
  border: 1px solid var(--md-typeset-table-color);
  padding: 12px;
}

.md-typeset__table tr:nth-child(2n) {
  background-color: var(--md-default-bg-color--lightest);
}
</style>

Detailed commands for reference:

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [8 x MI300X](https://docs.sglang.ai/references/amd.html#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)
- [16 x A100 (int8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (int8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)

### Download Weights
If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded. Please refer to [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) official guide to download the weights.

### Caching `torch.compile`
The DeepSeek series have huge model weights, it takes some time to compile the model with `torch.compile` for the first time if you have added the flag `--enable-torch-compile`. You can refer [here](https://docs.sglang.ai/backend/hyperparameter_tuning.html#try-advanced-options) to optimize the caching of compilation results, so that the cache can be used to speed up the next startup.

### Launch with one node of 8 x H200
Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended). **Note that Deepseek V3 is already in FP8. So we should not run it with any quantization arguments like `--quantization fp8 --kv-cache-dtype fp8_e5m2`.

### Running examples on Multi-node

- [Serving with two H20*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes).

- [Serving with two H200*8 nodes and docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker).

- [Serving with four A100*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes).

## Optimizations

### Multi-head Latent Attention (MLA) Throughput Optimizations

**Description**: [MLA](https://arxiv.org/pdf/2405.04434) is an innovative attention mechanism introduced by the DeepSeek team, aimed at improving inference efficiency. SGLang has implemented specific optimizations for this, including:

- **Weight Absorption**: By applying the associative law of matrix multiplication to reorder computation steps, this method balances computation and memory access and improves efficiency in the decoding phase.

- **MLA Attention Backends**: Currently SGLang supports different optimized MLA attention backends, including [FlashAttention3](https://github.com/Dao-AILab/flash-attention), [Flashinfer](https://docs.flashinfer.ai/api/mla.html), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [CutlassMLA](https://github.com/sgl-project/sglang/pull/5390), and [Triton](https://github.com/triton-lang/triton) backends. The default FA3 provides good performance across wide workloads.

- **FP8 Quantization**: W8A8 FP8 and KV Cache FP8 quantization enables efficient FP8 inference. Additionally, we have implemented Batched Matrix Multiplication (BMM) operator to facilitate FP8 inference in MLA with weight absorption.

- **CUDA Graph & Torch.compile**: Both MLA and Mixture of Experts (MoE) are compatible with CUDA Graph and Torch.compile, which reduces latency and accelerates decoding speed for small batch sizes.

- **Chunked Prefix Cache**: Chunked prefix cache optimization can increase throughput by cutting prefix cache into chunks, processing them with multi-head attention and merging their states. Its improvement can be significant when doing chunked prefill on long sequences. Currently this optimization is only available for FlashAttention3 backend.

Overall, with these optimizations, we have achieved up to **7x** acceleration in output throughput compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="Multi-head Latent Attention for DeepSeek Series Models">
</p>

**Usage**: MLA optimization is enabled by default.

**Reference**: Check [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [Slides](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf) for more details.

### Data Parallelism Attention

**Description**: This optimization involves data parallelism (DP) for the MLA attention mechanism of DeepSeek Series Models, which allows for a significant reduction in the KV cache size, enabling larger batch sizes. Each DP worker independently handles different types of batches (prefill, decode, idle), which are then synchronized before and after processing through the Mixture-of-Experts (MoE) layer. If you do not use DP attention, KV cache will be duplicated among all TP ranks.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="Data Parallelism Attention for DeepSeek Series Models">
</p>

With data parallelism attention enabled, we have achieved up to **1.9x** decoding throughput improvement compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="Data Parallelism Attention Performance Comparison">
</p>

**Usage**:
- Append `--enable-dp-attention --tp 8 --dp 8` to the server arguments when using 8 H200 GPUs. This optimization improves peak throughput in high batch size scenarios where the server is limited by KV cache capacity. However, it is not recommended for low-latency, small-batch use cases.
- DP and TP attention can be flexibly combined. For example, to deploy DeepSeek-V3/R1 on 2 nodes with 8 H100 GPUs each, you can specify `--enable-dp-attention --tp 16 --dp 2`. This configuration runs attention with 2 DP groups, each containing 8 TP GPUs.

**Reference**: Check [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models).

### Multi Node Tensor Parallelism

**Description**: For users with limited memory on a single node, SGLang supports serving DeepSeek Series Models, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**Usage**: Check [here](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) for usage examples.

### Block-wise FP8

**Description**: SGLang implements block-wise FP8 quantization with two key optimizations:

- **Activation**: E4M3 format using per-token-per-128-channel sub-vector scales with online casting.

- **Weight**: Per-128x128-block quantization for better numerical stability.

- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library optimized for FP8 matrix multiplications.

**Usage**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is enabled by default on NVIDIA Hopper GPUs and disabled by default on other devices. DeepGEMM can also be manually turned off by setting the environment variable `SGL_ENABLE_JIT_DEEPGEMM=0`.

Before serving the DeepSeek model, precompile the DeepGEMM kernels using:
```bash
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```
The precompilation process typically takes around 10 minutes to complete.

### Multi-token Prediction
**Description**: SGLang implements DeepSeek V3 Multi-Token Prediction (MTP) based on [EAGLE speculative decoding](https://docs.sglang.ai/backend/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved by **1.8x** for batch size 1 and **1.5x** for batch size 32 respectively on H200 TP8 setting.

**Usage**:
Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --trust-remote-code --tp 8
```
- The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
- FlashAttention3 FlashMLA and Triton backend fully supports MTP usage. For FlashInfer backend (`--attention-backend flashinfer`) with speculative decoding,`--speculative-eagle-topk` parameter should be set to `1`. MTP support for the CutlassMLA backend is still under development.
- To enable DeepSeek MTP for large batch sizes (>32), there are some parameters should be changed (Reference [this discussion](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)):
  - Adjust `--max-running-requests` to a larger number. The default value is `32` for MTP. For larger batch sizes, you should increase this value beyond the default value.
  - Set `--cuda-graph-bs`. It's a list of batch sizes for cuda graph capture. The default captured batch sizes for speculative decoding is set [here](https://github.com/sgl-project/sglang/blob/49420741746c8f3e80e0eb17e7d012bfaf25793a/python/sglang/srt/model_executor/cuda_graph_runner.py#L126). You can include more batch sizes into it.


### Reasoning Content for DeepSeek R1

See [Separate Reasoning](https://docs.sglang.ai/backend/separate_reasoning.html).


### Function calling for DeepSeek Models

Add arguments `--tool-call-parser deepseekv3` and `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`(recommended) to enable this feature. For example (running on 1 * H20 node):

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-0324 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --tool-call-parser deepseekv3 --chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja
```

Sample Request:

```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324", "tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of an city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "Hows the weather like in Qingdao today"}]}'
```

Expected Response

```
{"id":"6501ef8e2d874006bf555bc80cddc7c5","object":"chat.completion","created":1745993638,"model":"deepseek-ai/DeepSeek-V3-0324","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"id":"0","index":null,"type":"function","function":{"name":"query_weather","arguments":"{\"city\": \"Qingdao\"}"}}]},"logprobs":null,"finish_reason":"tool_calls","matched_stop":null}],"usage":{"prompt_tokens":116,"total_tokens":138,"completion_tokens":22,"prompt_tokens_details":null}}

```
Sample Streaming Request:
```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324","stream":true,"tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of an city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "Hows the weather like in Qingdao today"}]}'
```
Expected Streamed Chunks (simplified for clarity):
```
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"city"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\":\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"Q"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"ing"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"dao"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":null}}], "finish_reason": "tool_calls"}
data: [DONE]
```
The client needs to concatenate all arguments fragments to reconstruct the complete tool call:
```
{"city": "Qingdao"}
```
Important Notes:
1. Use a lower `"temperature"` value for better results.
2. To receive more consistent tool call results, it is recommended to use `--chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja`. It provides an improved unified prompt.


### Thinking Budget for DeepSeek R1

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

Launch a server with `--enable-custom-logit-processor` flag on.

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --enable-custom-logit-processor
```

Define `ThinkingBudgetLogitProcessor`

```python
from typing import TYPE_CHECKING, Any

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


if TYPE_CHECKING:
    from torch import Tensor
    from sglang.srt.managers.schedule_batch import Req


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits: "Tensor", custom_param_list: list[dict[str, Any]]) -> "Tensor"
        for i, param_dict in enumerate(custom_param_list):
            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
                continue

            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201
```

Sample Request:

```python
import openai
from rich.pretty import pprint


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```

## FAQ

1. **Question**: What should I do if model loading takes too long and NCCL timeout occurs?

    **Answer**: You can try to add `--dist-timeout 3600` when launching the model, this allows for 1-hour timeout.
