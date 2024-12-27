# Quantization

`SGLang` support various quantization methods, including online dynamic quantization and offline quantization.

Please visit [here](https://huggingface.co/collections/neuralmagic) for some popular quantized LLMs on huggingface.

## Online Dynamic Quantization

To enable online dynamic quantization, you can simply specify `--quantization` in the command line. For example, if you want to enable `FP8` quantization for model `meta-llama/Meta-Llama-3.1-8B-Instruct`, you can launch the server with the following command:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

Our team is working on supporting more quantization methods with high priority. We will soon support other quantization methods including but not limited to `["awq", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf"]`

We also support quantization methods based on [torchao](https://github.com/pytorch/ao). You can simply specify `--torchao-config` in the command line to support this feature. For example, if you want to enable `int4wo-128` for model `meta-llama/Meta-Llama-3.1-8B-Instruct`, you can launch the server with the following command:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int4wo-128 \
    --port 30000 --host 0.0.0.0
```

We support the following quantization methods based on torchao `["int8dq", "int8wo", "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row", "int4wo-32", "int4wo-64", "int4wo-128", "int4wo-256"]`

Note: According to [this issue](https://github.com/sgl-project/sglang/issues/2219#issuecomment-2561890230), `"int8dq"` method currently has some bugs when using together with cuda graph capture. So we suggest to disable cuda graph capture when using `"int8dq"` method. Namely, please use the following command:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int8dq \
    --disable-cuda-graph \
    --port 30000 --host 0.0.0.0
```


## Offline Quantization

To do offline quantization for your model, firstly you need to install [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:

```bash
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

Note: If the model has already quantized offline, please **do not** add `--quantization` argument when starting the engine.


## Reference

- [quantization document of vllm](https://docs.vllm.ai/en/latest/quantization/fp8.html)

- [torchao](https://github.com/pytorch/ao)

- [llm-compressor](https://github.com/vllm-project/llm-compressor/)