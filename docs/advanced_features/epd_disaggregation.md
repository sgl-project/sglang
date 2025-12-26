# EPD Disaggregation

## Why and What is EPD Disaggregation?

In modern Vision-Language Model (VLM) inference, request execution naturally decomposes into three distinct stages: Encoder, Prefill, and Decode.
The Encoder stage performs vision preprocessing and ViT-based image encoding, which is highly compute-intensive but only required during request initialization. The Prefill stage processes the full multimodal input sequence to initialize the language model’s Key-Value (KV) cache, while the Decode stage is dominated by memory bandwidth and KV cache access for autoregressive token generation.

Existing deployments typically colocate these stages within a unified execution engine, or at best apply Prefill–Decode (PD) disaggregation. However, such designs still tightly couple vision encoding with language prefill, leading to inefficient resource utilization, limited scalability for image-heavy workloads, and suboptimal scheduling under load.

To address these challenges, we introduce Encoder–Prefill–Decode (EPD) Disaggregation in SGLang. EPD further separates vision encoding from language processing, enabling independent horizontal scaling of encoder servers, improved load balancing for multimodal requests, and seamless integration with existing PD disaggregation to form a fully decoupled three-tier inference architecture.

### Usage

You can launch a language-only model using `--language-only`, or an encoder-only model using `--encoder-only`.
When launching a language-only model, you must additionally specify the encoder service endpoints via `--encoder-urls`.

We support multiple encoder transfer backends, including zmq_to_scheduler, zmq_to_tokenizer, and mooncake (the default is zmq_to_scheduler). The backend can be selected using `--encoder-transfer-backend`.

#### Qwen VL

- EP Disaggregation

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

- EPD Disaggregation

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# prefill 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
# decode 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003
# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000

```
