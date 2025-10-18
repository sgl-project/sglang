# Quantization

SGLang supports various quantization methods, including offline quantization and online dynamic quantization.

Offline quantization loads pre-quantized model weights directly during inference. This is required for quantization methods
such as GPTQ and AWQ, which collect and pre-compute various statistics from the original weights using the calibration dataset.

Online quantization dynamically computes scaling parameters—such as the maximum/minimum values of model weights—during runtime.
Like NVIDIA FP8 training's [delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8) mechanism, online quantization calculates the appropriate scaling factors
on-the-fly to convert high-precision weights into a lower-precision format.

**Note: For better performance, usability and convenience, offline quantization is recommended over online quantization.**

If you use a pre-quantized model, do not add `--quantization` to enable online quantization at the same time.
For popular pre-quantized models, please visit [ModelCloud](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)
or [NeuralMagic](https://huggingface.co/collections/neuralmagic) collections on HF for some
popular quality validated quantized models. Quantized models must be validated via benchmarks post-quantization
to guard against abnormal quantization loss regressions.

## Offline Quantization

To load already quantized models, simply load the model weights and config. **Again, if the model has been quantized offline,
there's no need to add `--quantization` argument when starting the engine. The quantization method will be parsed from the
downloaded Hugging Face config. For example, DeepSeek V3/R1 models are already in FP8, so do not add redundant parameters.**

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

#### Using [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)

NVIDIA Model Optimizer (ModelOpt) provides advanced quantization techniques optimized for NVIDIA hardware. SGLang includes a streamlined workflow for quantizing models with ModelOpt and automatically exporting them for deployment.

##### Installation

First, install ModelOpt. You can either install it directly or as an optional SGLang dependency:

```bash
# Option 1: Install ModelOpt directly
pip install nvidia-modelopt

# Option 2: Install SGLang with ModelOpt support (recommended)
pip install sglang[modelopt]
```

##### Quantization and Export Workflow

SGLang provides an example script that demonstrates the complete ModelOpt quantization and export workflow:

```bash
# Quantize and export a model using ModelOpt FP8 quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp8 \
    --quantization-method modelopt_fp8

# For FP4 quantization
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
    modelopt_export_path="./exported_model",
    modelopt_checkpoint_save_path="./checkpoint.pth",  # optional, fake quantized checkpoint
    trust_remote_code=True,
)

load_config = LoadConfig()
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

Or using the Python API:

```python
import sglang as sgl

# Deploy exported ModelOpt quantized model
llm = sgl.Engine(
    model_path="./quantized_tinyllama_fp8",
    quantization="modelopt"
)

# Run inference
prompts = ["Hello, how are you?", "What is the capital of France?"]
sampling_params = sgl.SamplingParams(temperature=0.8, max_new_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt: {prompts[i]}")
    print(f"Output: {output.outputs[0].text}")
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

**Export-only Workflow**: If you have a pre-existing fake quantized ModelOpt checkpoint, you can export it directly:

```python
model_config = ModelConfig(
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    quantization="modelopt_fp8",
    modelopt_checkpoint_restore_path="./my_checkpoint.pth",
    modelopt_export_path="./exported_model",
)
```

##### Benefits of ModelOpt

- **Hardware Optimization**: Specifically optimized for NVIDIA GPU architectures
- **Advanced Quantization**: Supports cutting-edge FP8 and FP4 quantization techniques
- **Seamless Integration**: Automatic export to HuggingFace format for easy deployment
- **Calibration-based**: Uses calibration datasets for optimal quantization quality
- **Production Ready**: Enterprise-grade quantization with NVIDIA support

## Online Quantization

To enable online quantization, you can simply specify `--quantization` in the command line. For example, you can launch the server with the following command to enable `FP8` quantization for model `meta-llama/Meta-Llama-3.1-8B-Instruct`:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

Our team is working on supporting more online quantization methods. SGLang will soon support methods including but not limited to `["awq", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf"]`.

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

## Reference

- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor/)
- [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [Torchao: PyTorch Architecture Optimization](https://github.com/pytorch/ao)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/)
