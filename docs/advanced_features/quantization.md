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
- [Torchao: PyTorch Architecture Optimization](https://github.com/pytorch/ao)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/)
