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

### Encoder transfer with Mooncake

`--encoder-transfer-backend mooncake` controls **how encoder outputs are transferred** between encoder and language/prefill services. It is an encoder transfer option and can be used independently of the global multimodal embedding cache.

Example:

```bash
# encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend mooncake \
  --port 30000

# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --encoder-transfer-backend mooncake \
  --port 30002
```

### Global multimodal embedding cache with Mooncake

SGLang also supports a Mooncake-backed **global multimodal embedding cache** for EPD workloads. When enabled on encoder servers, repeated image inputs can reuse previously computed ViT embeddings across instances instead of running the vision encoder again.

This feature is useful when:

- the deployment serves repeated or overlapping image inputs,
- encoder compute is the bottleneck, and
- Mooncake is already available in the cluster.

At a high level, the encoder checks whether the image embedding already exists in Mooncake. Cache hits are prefetched from the global store, while misses are encoded normally and inserted into the cache in the background.

To enable it:

- install and configure Mooncake in the same way as other SGLang Mooncake integrations,
- add `--enable-mm-global-cache` on the encoder server.

`--enable-mm-global-cache` controls **whether multimodal embeddings are looked up and stored in the global Mooncake cache**. It is separate from `--encoder-transfer-backend`, which only controls encoder output transport.

For Mooncake deployment and configuration details, see [HiCache best practices](hicache_best_practices.md#deployment-with-mooncake) and the [Mooncake backend README](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md).

Example:

```bash
# Shared Mooncake configuration
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_MASTER="127.0.0.1:50051"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb"

# encoder with global multimodal cache enabled
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --enable-mm-global-cache \
  --port 30000

# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --port 30002
```

Notes:

- This cache is for **multimodal encoder embeddings**, not the language model KV cache.
- The feature currently uses Mooncake as the shared backing store.
- It can be enabled regardless of which `--encoder-transfer-backend` you use.
- It is most relevant for EPD or encoder-disaggregated VLM deployments where the same images are likely to appear across requests or instances.

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

### Dynamic Encoder Discovery via Bootstrap Server

Instead of statically listing encoder URLs via `--encoder-urls`, you can have encoders register themselves at runtime. This is useful when encoders start after the prefill server, or when you need to add/replace encoders without restarting the prefill server.

Start the prefill/language-only server with `--encoder-bootstrap-port` to launch a built-in `EncoderBootstrapServer`. Encoder servers then use `--encoder-register-url` to register with it on startup.

```bash
# Step 1: Start the language-only prefill server with a local bootstrap server on port 8765
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-bootstrap-port 8765 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# Step 2: Start encoders — they self-register with the bootstrap server
# Encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-url http://127.0.0.1:8765 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Encoder 1 (can be added later without restarting the prefill server)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-url http://127.0.0.1:8765 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
```

You can also query the registered encoder URLs at any time:

```bash
curl http://127.0.0.1:30002/list_encoder_urls
# {"encoder_urls": ["http://127.0.0.1:30000", "http://127.0.0.1:30001"]}
```

If you prefer to run the bootstrap server on a separate host (e.g. shared across multiple prefill instances), use `--encoder-bootstrap-url` to point to it instead:

```bash
# External bootstrap server (started separately)
# e.g. reuse any existing EncoderBootstrapServer or start a dedicated one

# Prefill server using an external bootstrap
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-bootstrap-url http://bootstrap-host:8765 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

#### gRPC Encoder (EPD)

You can run the encoder as a gRPC server while keeping prefill/decode as HTTP.
When using gRPC encoders, set `SGLANG_ENCODER_MM_RECEIVER_MODE=grpc` for the
prefill process so it uses the gRPC receiver.

```bash
# gRPC encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --grpc-mode \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# prefill (HTTP) - tell it to use gRPC receiver
SGLANG_ENCODER_MM_RECEIVER_MODE=grpc \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls grpc://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# decode (HTTP)
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
