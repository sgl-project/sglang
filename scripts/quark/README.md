# Quantizing Sharded Grok-1 with Quark for SGLang

## Introduction

This guide explains how to use Quark to perform INT4-FP8 quantization on a sharded Grok-1 model, producing a quantized sharded checkpoint that can be loaded and executed in SGLang.

## Installation

1. Download [Quark](https://amd.com/bin/public/amdOpenDownload?filename=amd_quark-0.7%2080c639ab2c.zip) and extract the archive to access the Python wheel package and examples folder.

2. Install Quark using:
```bash
pip install amd-quark*.whl --ignore-requires-python
```

3. Install the additional dependencies
```bash
pip install -r requirements.txt
```

## Quantization

This guide supports two types of quantization:

1. Weight-only quantization
2. Weight, activation, and KV cache quantization

Refer to the following sections for detailed instructions on each method.

### INT4-FP8 Weight only 

#### Quick Start

Run the following command, replacing placeholders with the appropriate paths:

```bash
python quantize_sharded.py \
    --input <path_to_original_sharded_model_ckpt> \
    --output <path_to_output_quantized_sharded_model_ckpt> \
    --weight-only
```

Notes:
- Input model must be stored locally. For example, download https://huggingface.co/lmzheng/grok-1 to your local directory and specify its path using the `--input` argument.
- The final quantized and sharded checkpoint is saved at the path provided to --output.
- The entire script takes a few minutes to complete.

#### Quantization Stragegy

This model was created by applying Quark with calibration samples from Pile dataset.

- **Quantized Layers**: All linear layers excluding "lm_head", "*.gate"
- **Weight**: FP8 symmetric per-tensor, additionally, INT4 symmetric per-channel for MoE linear

#### INT4 Packing
Every eight `int4` values are packed into a single int32 integeter following the sequence defined by `order_map = [0, 2, 4, 6, 1, 3, 5, 7]`.

### INT4-FP8 Weight, FP8 Activation and FP8 KV Cache

Run the following command, replacing placeholders with the appropriate paths:

```bash
python quantize_sharded.py \
    --input <path_to_original_sharded_model_ckpt> \
    --output <path_to_output_quantized_sharded_model_ckpt> \
    --intermediate <path_to_store_intermediate_unsharded_model> \
    --quark-examples-dir <path_to_quark_examples>
```

Notes:
- Quark's LLM quantization is built on PyTorch and 🤗 Transformers, which currently do not support direct loading of sharded Grok-1 checkpoints.
- To work around this, the script first merges the sharded checkpoints into a single unsharded model, stored in the path specified by `--intermediate`.
- Input model must be stored locally. For example, download https://huggingface.co/lmzheng/grok-1 to your local directory and specify its path using the `--input` argument.
- The final quantized and sharded checkpoint is saved at the path provided to --output.
- The entire script takes 1~2 hours to complete.

#### Quantization Stragegy

This model was created by applying Quark with calibration samples from Pile dataset.

- **Quantized Layers**: All linear layers excluding "lm_head", "*.gate"
- **Weight**: FP8 symmetric per-tensor, additionally, INT4 symmetric per-channel for MoE linear
- **Activation**: FP8 symmetric per-tensor
- **KV Cache**: FP8 symmetric per-tensor

#### INT4 Packing
Every eight `int4` values are packed into a single int32 integeter following the sequence defined by `order_map = [0, 2, 4, 6, 1, 3, 5, 7]`.
