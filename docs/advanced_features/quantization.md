# Quantization

SGLang supports various quantization methods, including offline quantization and online dynamic quantization.

Offline quantization loads pre-quantized model weights directly during inference. This is required for quantization methods
such as GPTQ and AWQ, which collect and pre-compute various statistics from the original weights using the calibration dataset.

Online quantization dynamically computes scaling parameters—such as the maximum/minimum values of model weights—during runtime.
Like NVIDIA FP8 training's [delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8) mechanism, online quantization calculates the appropriate scaling factors
on-the-fly to convert high-precision weights into a lower-precision format.

**Note: For better performance, usability and convenience, offline quantization is recommended over online quantization.**

If you use a pre-quantized model, do not add `--quantization` to enable online quantization at the same time.
For popular pre-quantized models, please visit [Unsloth](https://huggingface.co/unsloth), [NVIDIA ModelOpt](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)
or [NeuralMagic](https://huggingface.co/collections/neuralmagic) collections on HF for some
popular quality validated quantized models. Quantized models must be validated via benchmarks post-quantization
to guard against abnormal quantization loss regressions.

## Platform Compatibility

The following table summarizes quantization method support across NVIDIA and AMD GPUs, Ascend NPUs.

| Method | NVIDIA GPUs | AMD GPUs (MI300X/MI325X/MI350X) | Ascend NPUs (A2/A3) | Notes |
|--------|:-----------:|:-------------------------------:|:-----------------------:|-------|
| `fp8` | Yes | Yes | WIP | Aiter or Triton backend on AMD |
| `mxfp4` | Yes | Yes | WIP | Requires CDNA3/CDNA4 with MXFP support; uses Aiter |
| `blockwise_int8` | Yes | Yes | No | Triton-based, works on both platforms |
| `w8a8_int8` | Yes | Yes | No | |
| `w8a8_fp8` | Yes | Yes | No | Aiter or Triton FP8 on AMD |
| `awq` | Yes | Yes | Yes | Uses Triton dequantize on AMD (vs. optimized CUDA kernels on NVIDIA). Uses CANN kernels on Ascend|
| `gptq` | Yes | Yes | Yes | Uses Triton or vLLM kernels on AMD. Uses CANN kernels on Ascend|
| `compressed-tensors` | Yes | Yes | Partial | Aiter paths for FP8/MoE on AMD. Uses CANN kernels on Ascend, `FP8` not supported yet|
| `quark` | Yes | Yes | No | AMD Quark quantization; Aiter GEMM paths on AMD |
| `auto-round` | Yes | Yes | Partial | Platform-agnostic (Intel auto-round). Uses CANN kernels on Ascend|
| `quark_int4fp8_moe` | No | Yes | No | AMD-only; online INT4-to-FP8 MoE quantization (CDNA3/CDNA4) |
| `awq_marlin` | Yes | No | No | Marlin kernels are CUDA-only |
| `gptq_marlin` | Yes | No | No | Marlin kernels are CUDA-only |
| `gguf` | Yes | No | Yes | CUDA-only kernels in sgl-kernel; Pre-dequantized on Ascend |
| `modelopt` / `modelopt_fp8` | Yes (Hopper/SM90+) | No | No | [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer); requires NVIDIA hardware |
| `modelopt_fp4` | Yes (Blackwell/SM100+) | No | No | [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer); native FP4 on Blackwell (B200, GB200) |
| `petit_nvfp4` | No | Yes (MI250/MI300X/MI325X) | No | Enables NVFP4 on ROCm via [Petit](https://github.com/causalflow-ai/petit-kernel); use `modelopt_fp4` on NVIDIA Blackwell. Auto-selected when loading NVFP4 models on AMD. See [LMSYS blog](https://lmsys.org/blog/2025-09-21-petit-amdgpu/) and [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html). |
| `bitsandbytes` | Yes | Experimental | No | Depends on bitsandbytes ROCm support |
| `torchao` (`int4wo`, etc.) | Yes | Partial | No | `int4wo` not supported on AMD; other methods may work |
| `modelslim` | No | No | Yes | Ascend quantization; Uses CANN kernels |

On AMD, several of these methods use [Aiter](https://github.com/ROCm/aiter) for acceleration -- set `SGLANG_USE_AITER=1` where noted. See [AMD GPU setup](../platforms/amd_gpu.md) for installation and configuration details.

On Ascend, various layers quantization configurations are supported, see [Ascend NPU quantization](../platforms/ascend/ascend_npu_quantization.md) for details.

## GEMM Backends for FP4/FP8 Quantization

:::{note}
Backend selection is supported only for **blockwise FP8** and **NVFP4** GEMM. When running FP8 or FP4 quantized models, you can select the GEMM backend via `--fp8-gemm-backend` and `--fp4-gemm-backend`.
:::

### `--fp8-gemm-backend` (Blockwise FP8 GEMM)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | All | Auto-selects based on hardware |
| `deep_gemm` | SM90, SM100 | JIT-compiled; enabled when DeepGEMM is installed |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM backend; optimal for low-latency |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS groupwise FP8 GEMM |
| `flashinfer_deepgemm` | SM90 | Uses swapAB optimization for small M dimensions in decoding |
| `cutlass` | SM90, SM100/120 | sgl-kernel CUTLASS |
| `triton` | All | Fallback; widely compatible |
| `aiter` | ROCm | AMD AITER backend |

**`auto` selection order:** 1) DeepGEMM (SM90/SM100, installed); 2) FlashInfer TRTLLM (SM100, FlashInfer available); 3) CUTLASS (SM90/SM100/120); 4) AITER (AMD); 5) Triton. **Exception:** SM120 always resolves to Triton.

### `--fp4-gemm-backend` (NVFP4 GEMM)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | SM100/120 | Auto-selects: `flashinfer_cudnn` on SM120; `flashinfer_cutlass` on SM100 |
| `cutlass` | SM100/120 | SGLang CUTLASS kernel |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS backend |
| `flashinfer_cudnn` | SM100/120 (CUDA 13+, cuDNN 9.15+) | FlashInfer cuDNN backend; used on SM120 for performance |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM backend |

When FlashInfer is unavailable for NVFP4, the SGLang CUTLASS kernel is used as an automatic fallback.

## Offline Quantization

To load already quantized models, simply load the model weights and config. **Again, if the model has been quantized offline,
there's no need to add `--quantization` argument when starting the engine. The quantization method will be parsed from the
downloaded Hugging Face or msModelSlim config. For example, DeepSeek V3/R1 models are already in FP8, so do not add redundant parameters.**

```bash
python3 -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --port 30000 --host 0.0.0.0
```

Take note, if your model is **per-channel quantized (INT8 or FP8) with per-token dynamic quantization activation**, you can opt to include `--quantization w8a8_int8` or `--quantization w8a8_fp8` to invoke the corresponding CUTLASS int8_kernel or fp8_kernel in sgl-kernel. This action will ignore the Hugging Face config's quantization settings. For instance, with `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`, if you execute with `--quantization w8a8_fp8`, the system will use the `W8A8Fp8Config` from SGLang to invoke the sgl-kernel, rather than the `CompressedTensorsConfig` for vLLM kernels.

```bash
python3 -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
    --quantization w8a8_fp8 \
    --port 30000 --host 0.0.0.0
```

### Examples of Offline Model Quantization

#### Using [Unsloth](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)

We strongly suggest the use of Unsloth to quantize and load the model. Please refer to [SGLang Deployment & Inference Guide with Unsloth](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide).

#### Using [auto-round](https://github.com/intel/auto-round)

```bash
# Install
pip install auto-round
```

- LLM quantization

```py
# for LLM
from auto_round import AutoRound
model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-autoround-4bit"
# Scheme examples: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
scheme = "W4A16"
format = "auto_round"
autoround = AutoRound(model_id, scheme=scheme)
autoround.quantize_and_save(quant_path, format=format) # quantize and save

```

- VLM quantization
```py
# for VLMs
from auto_round import AutoRoundMLLM
model_name = "Qwen/Qwen2-VL-2B-Instruct"
quant_path = "Qwen2-VL-2B-Instruct-autoround-4bit"
scheme = "W4A16"
format = "auto_round"
autoround = AutoRoundMLLM(model_name, scheme)
autoround.quantize_and_save(quant_path, format=format) # quantize and save

```

- Command Line Usage (Gaudi/CPU/Intel GPU/CUDA)

```bash
auto-round \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

- known issues

Several limitations currently affect offline quantized model loading in sglang, These issues might be resolved in future updates of sglang. If you experience any problems, consider using Hugging Face Transformers as an alternative.

1. Mixed-bit Quantization Limitations

    Mixed-bit quantization is not fully supported. Due to vLLM's layer fusion (e.g., QKV fusion), applying different bit-widths to components within the same fused layer can lead to compatibility issues.


2. Limited Support for Quantized MoE Models

    Quantized MoE models may encounter inference issues due to kernel limitations (e.g., lack of support for mlp.gate layer quantization). please try to skip quantizing these layers to avoid such errors.


3. Limited Support for Quantized VLMs
    <details>
        <summary>VLM failure cases</summary>

    Qwen2.5-VL-7B

    auto_round:auto_gptq format:  Accuracy is close to zero.

    GPTQ format:  Fails with:
    ```
    The output size is not aligned with the quantized weight shape
    ```
    auto_round:auto_awq and AWQ format:  These work as expected.
    </details>

#### Using [GPTQModel](https://github.com/ModelCloud/GPTQModel)

```bash
# install
pip install gptqmodel --no-build-isolation -v
```

```py
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128) # quantization config
model = GPTQModel.load(model_id, quant_config) # load model

model.quantize(calibration_dataset, batch_size=2) # quantize
model.save(quant_path) # save model
```

#### Using [LLM Compressor](https://github.com/vllm-project/llm-compressor/)

```bash
# install
pip install llmcompressor
```

Here, we take quantize `meta-llama/Meta-Llama-3-8B-Instruct` to `FP8` as an example to elaborate on how to do offline quantization.

```python
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Step 1: Load the original model.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = SparseAutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Step 2: Perform offline quantization.
# Step 2.1: Configure the simple PTQ quantization.
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Step 2.2: Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Step 3: Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

Then, you can directly use the quantized model with `SGLang`, by using the following command:

```bash
python3 -m sglang.launch_server \
    --model-path $PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic \
    --port 30000 --host 0.0.0.0
```

#### Using [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)

NVIDIA Model Optimizer (ModelOpt) provides advanced quantization techniques optimized for NVIDIA hardware.

**Offline vs. Online Quantization:**

SGLang supports two modes for ModelOpt.

* **Offline Quantization (pre-quantized):**
    * **Usage:** Download a pre-quantized model from Hugging Face or run `hf_ptq.py` once to create a new quantized checkpoint. Then load this quantized checkpoint.
    * **Pros:** Fast server startup, quantization can be validated before deployment, efficient resource usage.
    * **Cons:** Requires an extra preparation step.

* **Online Quantization (quant and serve):**
    * **Usage:** Load a standard BF16/FP16 model and add a flag. The engine applies quantization *on startup*.
    * **Pros:** Convenient (no new checkpoint needed).
    * **Cons:** **High startup time**, increases VRAM usage during initialization (risk of OOM).

The following sections guide you through using the Offline path: loading pre-quantized models or creating your own checkpoints.

##### Using Pre-Quantized Checkpoints

If a model is already quantized (e.g., from Hugging Face), you can load it directly.

* **FP8 Models:**
    Use `--quantization modelopt_fp8`.
    ```bash
    python3 -m sglang.launch_server \
        --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
        --quantization modelopt_fp8 \
        --port 30000
    ```

* **FP4 Models:**
    Use `--quantization modelopt_fp4`.
    ```bash
    python3 -m sglang.launch_server \
        --model-path nvidia/Llama-3.3-70B-Instruct-NVFP4 \
        --quantization modelopt_fp4 \
        --port 30000
    ```

##### Creating Your Own Quantized Checkpoints

If a pre-quantized checkpoint is not available for your model, you can create one using NVIDIA Model Optimizer's `hf_ptq.py` script.

**Why quantize?**
- Reduce VRAM usage
- Higher throughput and lower latency
- More flexible deployment (on smaller GPUs)

**What can be quantized?**
- The entire model
- MLP layers only
- KV cache

**Key options in `hf_ptq.py`:**

`--qformat`: Quantization formats `fp8`, `nvfp4`, `nvfp4_mlp_only`

`--kv_cache_qformat`: KV cache quantization format (default: `fp8`)

**Note:** The default `kv_cache_qformat` may not be optimal for all use cases. Consider setting this explicitly.

**Hardware requirements:** Hopper and higher are recommended. Insufficient GPU memory may cause weight offloading, resulting in extremely long quantization time.

For detailed usage and supported model architectures, see [NVIDIA Model Optimizer LLM PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq).

SGLang includes a streamlined workflow for quantizing models with ModelOpt and automatically exporting them for deployment.

##### Installation

First, install ModelOpt:

```bash
pip install nvidia-modelopt
```

##### Quantization and Export Workflow

SGLang provides an example script that demonstrates the complete ModelOpt quantization and export workflow. Run from the SGLang repository root (see [modelopt_quantize_and_export.py](https://github.com/sgl-project/sglang/blob/main/examples/usage/modelopt_quantize_and_export.py)):

```bash
# Quantize and export a model using ModelOpt FP8 quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp8 \
    --quantization-method modelopt_fp8

# For FP4 quantization (requires Blackwell GPU)
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp4 \
    --quantization-method modelopt_fp4
```

##### Available Quantization Methods

- `modelopt_fp8`: FP8 quantization with optimal performance on NVIDIA Hopper and Blackwell GPUs
- `modelopt_fp4`: FP4 quantization with optimal performance on Nvidia Blackwell GPUs

##### Python API Usage

You can also use ModelOpt quantization programmatically:

```python
import sglang as sgl
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader

# Configure model with ModelOpt quantization and export
model_config = ModelConfig(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization="modelopt_fp8",  # or "modelopt_fp4"
    trust_remote_code=True,
)

load_config = LoadConfig(
    modelopt_export_path="./exported_model",
    modelopt_checkpoint_save_path="./checkpoint.pth",  # optional, fake quantized checkpoint
)
device_config = DeviceConfig(device="cuda")

# Load and quantize the model (export happens automatically)
model_loader = get_model_loader(load_config, model_config)
quantized_model = model_loader.load_model(
    model_config=model_config,
    device_config=device_config,
)
```

##### Deploying Quantized Models

After quantization and export, you can deploy the model with SGLang:

```bash
# Deploy the exported quantized model
python -m sglang.launch_server \
    --model-path ./quantized_tinyllama_fp8 \
    --quantization modelopt \
    --port 30000 --host 0.0.0.0
```

Or using the Python API (use the same path as `modelopt_export_path` from the quantize step):

```python
import sglang as sgl

def main():
    # Deploy exported ModelOpt quantized model
    # Path must match modelopt_export_path from quantize step (e.g., ./exported_model)
    llm = sgl.Engine(
        model_path="./exported_model",
        quantization="modelopt",
    )

    # Run inference
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 100,
    }

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output['text']}")

if __name__ == "__main__":
    main()

```

##### Advanced Features

**Checkpoint Management**: Save and restore fake quantized checkpoints for reuse:

```bash
# Save the fake quantized checkpoint during quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --export-dir ./quantized_model \
    --quantization-method modelopt_fp8 \
    --checkpoint-save-path ./my_checkpoint.pth

# The checkpoint can be reused for future quantization runs and skip calibration
```

**Export-only Workflow**: If you have a pre-existing fake quantized ModelOpt checkpoint, you can export it directly. See [LoadConfig](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/load_config.py) for the full API:

```python
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader

model_config = ModelConfig(
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    quantization="modelopt_fp8",
    trust_remote_code=True,
)

load_config = LoadConfig(
    modelopt_checkpoint_restore_path="./my_checkpoint.pth",
    modelopt_export_path="./exported_model",
)

# Load and export the model (DeviceConfig defaults to device="cuda")
model_loader = get_model_loader(load_config, model_config)
model_loader.load_model(model_config=model_config, device_config=DeviceConfig())
```

##### Benefits of ModelOpt

- **Hardware Optimization**: Specifically optimized for NVIDIA GPU architectures
- **Advanced Quantization**: Supports cutting-edge FP8 and FP4 quantization techniques
- **Seamless Integration**: Automatic export to HuggingFace format for easy deployment
- **Calibration-based**: Uses calibration datasets for optimal quantization quality
- **Production Ready**: Enterprise-grade quantization with NVIDIA support

#### Using [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
MindStudio-ModelSlim (msModelSlim) is a model offline quantization compression tool launched by MindStudio and optimized for Ascend hardware.

- **Installation**

    ```bash
    # Clone repo and install msmodelslim:
    git clone https://gitcode.com/Ascend/msmodelslim.git
    cd msmodelslim
    bash install.sh
    ```

- **LLM quantization**

    Download the original floating-point weights of the large model. Taking Qwen3-32B as an example, you can go to [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) to obtain the original model weights. Then install other dependencies (related to the model, refer to the huggingface model card).
    > Note: You can find pre-quantized validated models on [modelscope/Eco-Tech](https://modelscope.cn/models/Eco-Tech).

    _Traditional quantification methods require the preparation of calibration data files (```.jsonl``` formats) for calibration in the quantification process._
    ```bash
    Qwen3-32B/      # floating-point model downloaded from official HF (or modelscope) repo
    msmodelslim/    # msmodelslim repo
      |----- lab_calib # calibration date folder (put your dataset here in ```.jsonl``` format or use pre-prepared ones)
          |----- some file (such as laos_calib.jsonl)
      |----- lab_practice # best practice folder with configs for quantization
          |----- model folder (such as qwen3_5_moe folder) # folder with quantization configs
              |----- quant_config (such as qwen3_5_moe_w8a8.yaml) # quantization config
      |----- another folders
    output_folder/   # generated by below command
      |----- quant_model_weights-00001-of-0001.safetensors # quantized weights
      |----- quant_model_description.json # file with description of the quantization methods for each layer (```W4A4_DYNAMIC```, etc.)
      |----- another files (such as config.json, tokenizer.json, etc.)
    ```
    Run quantization using one-click quantization (recommended):
    ```bash
    msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu:0,1 \
    --model_type Qwen3-32B \
    --quant_type w8a8 \
    --trust_remote_code True
    ```

- **Usage Example**
    ```bash
    python3 -m sglang.launch_server \
    --model-path $PWD/Qwen3-32B-w8a8 \
    --port 30000 --host 0.0.0.0
    ```

- **Available Quantization Methods**:
    - [x]  ```W4A4_DYNAMIC``` linear with online quantization of activations
    - [x]  ```W8A8``` linear with offline quantization of activations
    - [x]  ```W8A8_DYNAMIC``` linear with online quantization of activations
    - [x]  ```W4A4_DYNAMIC``` MOE with online quantization of activations
    - [x]  ```W4A8_DYNAMIC``` MOE with online quantization of activations
    - [x]  ```W8A8_DYNAMIC``` MOE with online quantization of activations
    - [ ]  ```W4A8``` linear TBD
    - [ ]  ```W4A16``` linear TBD
    - [ ]  ```W48A16``` linear TBD
    - [ ]  ```W4A16``` MoE in progress
    - [ ]  ```W8A16``` MoE in progress
    - [ ]  ```KV Cache``` in progress
    - [ ]  ```Attention``` in progress


For more detailed examples of quantization of models, as well as information about their support, see the [examples](https://gitcode.com/Ascend/msmodelslim/blob/master/example/README.md) section in ModelSLim repo.

## Online Quantization

To enable online quantization, you can simply specify `--quantization` in the command line. For example, you can launch the server with the following command to enable `FP8` quantization for model `meta-llama/Meta-Llama-3.1-8B-Instruct`:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

Our team is working on supporting more online quantization methods. SGLang will soon support methods including but not limited to `["awq", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf"]`.

### torchao online quantization method

SGLang also supports quantization methods based on [torchao](https://github.com/pytorch/ao). You can simply specify `--torchao-config` in the command line to support this feature. For example, if you want to enable `int4wo-128` for model `meta-llama/Meta-Llama-3.1-8B-Instruct`, you can launch the server with the following command:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int4wo-128 \
    --port 30000 --host 0.0.0.0
```

SGLang supports the following quantization methods based on torchao `["int8dq", "int8wo", "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row", "int4wo-32", "int4wo-64", "int4wo-128", "int4wo-256"]`.

Note: According to [this issue](https://github.com/sgl-project/sglang/issues/2219#issuecomment-2561890230), `"int8dq"` method currently has some bugs when using together with cuda graph capture. So we suggest to disable cuda graph capture when using `"int8dq"` method. Namely, please use the following command:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int8dq \
    --disable-cuda-graph \
    --port 30000 --host 0.0.0.0
```

### `quark_int4fp8_moe` online quantization method

SGLang running on AMD GPUs (CDNA3 or CDNA4 architecture) supports the quantization method `--quantization quark_int4fp8_moe`, that will replace [MoE layers](https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L271) originally in high precision (bfloat16, float16 or float32) to use weights dynamically quantized to int4, that are upcasted to float8 during inference to run compute in float8 precision with activations dynamically quantized on the fly to float8.

Other layers (e.g. projections in the attention layers) have their weights quantized online to float8 directly.

## Reference

- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor/)
- [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Model Optimizer LLM PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq)
- [Petit: NVFP4 on ROCm](https://github.com/causalflow-ai/petit-kernel) — [LMSYS blog](https://lmsys.org/blog/2025-09-21-petit-amdgpu/), [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html)
- [Torchao: PyTorch Architecture Optimization](https://github.com/pytorch/ao)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/)
- [auto-round](https://github.com/intel/auto-round)
- [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
