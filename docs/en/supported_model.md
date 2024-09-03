# Supported Models

SGLang supports a wide range of generative and embedding models. Below is a comprehensive list of supported models along with specific serving commands where applicable.

## Generative Models

### Language Models
- Llama / Llama 2 / Llama 3 / Llama 3.1
- Mistral / Mixtral / Mistral NeMo
- Gemma / Gemma 2
- Qwen / Qwen 2 / Qwen 2 MoE
- DeepSeek / DeepSeek 2
- StableLM
- Command-R
- DBRX
- Grok
- ChatGLM
- InternLM 2
- Exaone 3

### Vision-Language Models
1. [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
   ```bash
   # For 7B model
   python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --port=30000 --chat-template=chatml-llava
   
   # For 72B model
   python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-72b-ov --port=30000 --tp-size=8 --chat-template=chatml-llava
   ```

2. LLaVA 1.5 / 1.6 / NeXT
   ```bash
   # For 8B model
   python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000 --tp-size=1 --chat-template=llava_llama_3
   
   # For 72B model
   python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --port=30000 --tp-size=8 --chat-template=chatml-llava
   ```

3. Yi-VL

**Note**: Query vision-language model servers using the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test_vision_openai_server.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server.py)

## Embedding Models

1. e5-mistral
2. gte-Qwen2
   ```bash
   python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding
   ```

## Using Models from ModelScope

To use a model from [ModelScope](https://www.modelscope.cn):

1. Set the environment variable:
   ```bash
   export SGLANG_USE_MODELSCOPE=true
   ```

2. Launch the server (e.g., for Qwen2-7B-Instruct):
   ```bash
   SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
   ```

## Running Large Models: Llama 3.1 405B

### Single Node (FP8)
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

### Two Nodes (FP16)
Replace `172.16.4.52:20000` with your first node's IP address and port.

Node 1:
```bash
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0 --disable-cuda-graph
```

Node 2:
```bash
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 1 --disable-cuda-graph
```

For instructions on supporting new models, refer to the model support documentation in the next page.