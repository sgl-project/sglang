## Tuning Triton MoE Kernels

This directory contains benchmarking tools for MoE (Mixture of Experts) kernels.

### Overview

The tuning tools support both **Tensor Parallelism (TP)** and **Expert Parallelism (EP)** modes:

- **TP Mode**: Traditional tensor parallelism where intermediate layers are sharded across GPUs
- **EP Mode**: Expert parallelism where experts are distributed across GPUs. Can be combined with TP mode (e.g., `--tp-size 8 --ep-size 2`)
- **MLLM Support**: Multi-modal Large Language Models with text encoders (e.g., Llama4, Qwen3VL)

### Tuning Tools

#### 1. `tuning_fused_moe_triton.py`
A unified tool for tuning the `fused_moe_triton` kernel. Adapted from [vllm's benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py), with support for EP mode and various model architectures.

#### 2. `tuning_fused_moe_triton_sep.py`
A specialized tool for separate kernel tuning, optimizing the first and second MoE kernels independently with TMA (Tensor Memory Accelerator) support.

### Usage Examples

#### User-facing Auto-tune Entrypoint
```bash
# Inspect the resolved model/kernel tuning plan without running GPU benchmarks.
PYTHONPATH=$PWD/python:$PWD:$PYTHONPATH \
python -m sglang.auto_tune \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 1 \
    --quick \
    --dry-run \
    --output-dir /tmp/sglang-moe-configs

# Run a reduced tuning pass and write configs under
# /tmp/sglang-moe-configs/configs/triton_<version>/.
PYTHONPATH=$PWD/python:$PWD:$PYTHONPATH \
python -m sglang.auto_tune \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 1 \
    --quick \
    --batch-size 1,8,32,128 \
    --output-dir /tmp/sglang-moe-configs

# Reuse the generated configs when launching SGLang.
SGLANG_MOE_CONFIG_DIR=/tmp/sglang-moe-configs \
python -m sglang.launch_server \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 1
```

The `python -m sglang.auto_tune` entrypoint currently wraps the unified
`fused_moe_triton` tuner in this directory. It resolves the model MoE shape,
chooses the existing SGLang config filename convention, writes configs in the
same directory layout read by `SGLANG_MOE_CONFIG_DIR`, and prints a final reuse
command. Use `--quick` for a smaller candidate search before running a full
tuning pass. After tuning, the entrypoint validates that the generated JSON is
loadable through the same runtime `SGLANG_MOE_CONFIG_DIR` path used by
`get_moe_configs`, which catches misplaced config directories before serving.
The underlying tuner uses Ray workers, so install SGLang's Ray extra or run
`pip install 'ray[default]>=2.54.0'` if your environment does not already
provide Ray.

#### Basic TP Mode Tuning
```bash
# Tune Mixtral-8x7B with default TP settings
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tune

# Tune Qwen2-57B with FP8 and TP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --dtype fp8_w8a8 \
    --tune

# Tune DeepSeek-V3 with FP8 and TP=8
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --tune
```

#### EP Mode Tuning (Expert Parallelism)
**Note**: EP mode can be used alone or combined with TP mode. When using both, ensure `tp_size` is divisible by `ep_size`.

```bash
# Tune Mixtral-8x7B with EP=2 only
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --ep-size 2 \
    --tune

# Tune Qwen2-57B with TP=8 and EP=4 (combined mode)
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 8 \
    --ep-size 4 \
    --dtype fp8_w8a8 \
    --tune
```

#### MLLM Model Tuning (Multi-modal)
```bash
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tp-size 2 \
    --tune
```

#### Separate Kernel Tuning with `tuning_fused_moe_triton_sep.py`

This tool requires pre-generated topk_ids files and supports both TP and EP modes:

Edit the code file (such as srt/models/deepseek_v2.py) in the Python site package and add the logic for saving topk_ids:

```python
# import get_tensor_model_parallel_rank
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] >= 4096 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(topk_output.topk_ids, f"{topk_ids_dir}/topk_ids_layer{self.layer_id}_idx{self.save_idx}.pt")
    self.save_idx += 1
```

Launch sglang server and send request using `benchmark/kernels/fused_moe_triton/tuning_client.py`
```bash
python benchmark/kernels/fused_moe_triton/tuning_client.py --port 8000
```

```bash
# TP Mode: Tune separate kernels with TP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# EP Mode: Tune separate kernels with TP=4 and EP=2
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 4 \
    --ep-size 2 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# MLLM: Tune DeepSeek-V3 with separate kernels, TP=8 and EP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8 \
    --ep-size 4 \
    --dtype fp8_w8a8 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# Benchmark specific config without tuning
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 4 \
    --batch-size 1024 \
    --dtype fp8_w8a8 \
    --configs 128 256 128 16 8 4 \
    --topk-ids-dir /path/to/topk_ids
```

#### Advanced Options
```bash
# Channel-wise quantization
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model meituan/DeepSeek-R1-Channel-INT8 \
    --tp-size 16 \
    --dtype int8_w8a8 \
    --per-channel-quant \
    --tune

# Specific batch size tuning
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 2048 \
    --tune
```

### Configuration Files

After tuning, configuration files will be generated:
- **Standard tuning**: `E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json`
- **Separate kernel tuning**: Two files for up/down kernels with TMA optimization flags

Move these files to `sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_version/` directory to use them in SGLang.

### Supported Models

- **Mixtral**: mistralai/Mixtral-8x7B-Instruct-v0.1, mixtral-8x22b
- **Qwen**: Qwen2-57B, Qwen3-235B, Qwen3VL (MLLM)
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3, DeepSeek-R1
- **Llama**: Llama4-Vision (MLLM)
- **DBRX**: databricks/dbrx-instruct
- **Jamba**: ai21labs/AI21-Jamba
- **Grok**: xai-org/grok-1
- **GLM**: THUDM/glm-4-9b-chat
- **Bailing**: Custom MoE models

### Parameters Reference

- `--model`: HuggingFace model name or local path
- `--tp-size`: Tensor parallelism size (default: 2)
- `--ep-size`: Expert parallelism size (default: 1, can be combined with TP mode, ensure tp_size is divisible by ep_size)
- `--dtype`: Data type (`auto`, `fp8_w8a8`, `int8_w8a16`, `int8_w8a8`)
- `--batch-size`: Specific batch size for tuning (optional)
- `--tune`: Enable tuning mode
- `--per-channel-quant`: Enable per-channel quantization
- `--disable-shared-experts-fusion`: Disable shared expert fusion for some models
- `--topk-ids-dir`: Directory containing pre-generated topk_ids (for sep tool only)
- `--configs`: Manual config specification [BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages]

### Performance Comparison Tool

- `benchmark_vllm_vs_sglang_fused_moe_triton.py`: A tool for comparing the performance of fused MoE kernels between vllm and sglang implementations. Supports various model architectures and data types.

Example usage:
```bash
# Compare with default settings (Mixtral model)
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py

# Compare with FP8 mode for Qwen2-57B
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --use-fp8-w8a8

# Compare with custom TP size
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8

# Compare with custom TP size
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8
```

The benchmark results will be saved as plots and data files in the specified output directory (default: `./configs/benchmark_ops/vllm_sglang_fused_moe/`).

- `benchmark_torch_compile_fused_moe.py`: A tool for benchmarking the performance of the fused MoE kernel with `torch.compile` and original fused MoE kernel.

Usage is similar to `benchmark_vllm_vs_sglang_fused_moe_triton.py`, note that `torch.compile` does not support `fp8_w8a8` and `int8_w8a8` fused_moe_kernel. Both tools now support EP mode with `--ep-size` parameter.
