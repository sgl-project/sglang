# Quantization

`SGLang` support various quantization methods, inclding online dynamic quantization and offline quantization.

## Online Dynamic Quantization

To enable oneline dynamic quantization, you can simply specify `--quantize` in command line. For example, if you want to enable `FP8` quantization for model `meta-llama/Meta-Llama-3.1-8B-Instruct`, you can lauch the server with following command:
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

Then, you can test your FP8 quantized model with `curl`:
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -d '{"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

<details>
<summary>More quantization methods</summary>

Our team is wokring on supporting more quantization methods with high priority. We will soon support other quantization methods including but not limitied to `["awq", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf"]`

</details>

We also support quantization methods based on [torchao](https://github.com/pytorch/ao). You can simply specify `--torchao-config` in command line to support this feature. For example, if you want to enable `int4wo-128` for model `meta-llama/Meta-Llama-3.1-8B-Instruct`, you can lauch the server with following command:
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int4wo-128 \
    --port 30000 --host 0.0.0.0
``` 

<details>
<summary>More quantization methods based on torchao</summary>

We support the following quantization methods based on torchao `["int8dq", "int8wo", "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row", "int4wo-32", "int4wo-64", "int4wo-128", "int4wo-256"]`

Note: `"int8dq"` method currently has some bugs when using together with cuda graph capture. So we suggest to disable cuda graph capture when using `"int8dq"` method. Namely, please use the following command:
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int8dq \
    --disable-cuda-graph \
    --port 30000 --host 0.0.0.0
```

</details>


## Offline Quantization

To do offline quantization for your model, firstly you need to install [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:
```bash
pip install llmcompressor
```

Here, we take quantize `meta-llama/Meta-Llama-3-8B-Instruct` to `FP8` as an example to elaborate how to do offline quantization.
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

Then, you can directly use the quantized model with `SGLang`, using the following command:
```bash
python3 -m sglang.launch_server \
    --model-path $PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic \
    --port 30000 --host 0.0.0.0
```


## Reference

- [quantization document of vllm](https://docs.vllm.ai/en/latest/quantization/fp8.html)

- [torchao](https://github.com/pytorch/ao)

- [llm-compressor](https://github.com/vllm-project/llm-compressor/)



