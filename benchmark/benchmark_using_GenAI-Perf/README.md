# Benchmarking SGLang Using GenAI-Perf

GenAI-Perf is a powerful command-line tool used to evaluate the performance of generative AI models by assessing their throughput and latency when deployed via an inference server. For detailed instructions on using GenAI-Perf, please visit the [official GenAI-Perf documentation.](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md).


To benchmark SGLang using GenAI-Perf, follow these steps. This example uses the meta-llama/Llama-3.2-1B model; ensure you replace the model and tokenizer paths with your actual model and tokenizer locations. The meta-llama/Llama-3.2-1B model used in this example is available at [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

## 1. Run the SGLang Server

First, ensure the SGLang server is running with the desired model. For this example, we use the Llama-3.2-1B model. The command to launch the SGLang server is as follows:

```
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \ # Replace with your model path here
    --port 8000 \ # Replace with your port number here
    --trust-remote-code \
    --chat-template chatml
```

## 2. Using GenAI-Perf to benchmark

### Install GenAI-Perf
On Ubuntu 24.04, install GenAI-Perf using pip:
```
pip install genai-perf
```

For other platforms, refer to the [installation instructions.](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md#installation).

### Run GenAI-Perf
Next, use GenAI-Perf to benchmark the model served by SGLang. The following command specifies the model, tokenizer, service kind, endpoint type, and other benchmark parameters:

```
  genai-perf profile \
    -m Qwen/Qwen3-8B \ # Replace with your model name here
    --tokenizer /tmp/Qwen/Qwen3-8B/ # Replace with your tokenizer path here
    --endpoint-type chat \
    --request-count 1\
    --url http://localhost:8000 # Replace with your SGLang server URL here
```

## 3. Interpret the Results

GenAI-Perf prints a table on console with various metrics and also a results profile_export_genai_perf.json file, including:

- **Time to First Token (TTFT):** Time between when a request is sent and when its first response is received, one value per request in benchmark
- **Inter Token Latency (ITL):** Time between intermediate responses for a single request divided by the number of generated tokens of the latter response, one value per response per request in benchmark
- **Request Latency:** Time between when a request is sent and when its final response is received, one value per request in benchmark
- **Output Sequence Length:** 	Total number of output tokens of a request, one value per request in benchmark
- **Input Sequence Length:** Total number of input tokens of a request, one value per request in benchmark
- **Output Token Throughput:** Total number of output tokens from benchmark divided by benchmark duration
- **Request Throughput:** Number of final responses from benchmark divided by benchmark duration

These metrics provide insights into the performance and efficiency of the SGLang server with the Llama-3.2-1B model.


---

### Additional Resources

For more details about using GenAI-Perf, you can refer to the official documentation:
[GenAI-Perf Documentation](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md).
