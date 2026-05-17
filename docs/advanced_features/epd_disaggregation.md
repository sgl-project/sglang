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
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend mooncake \
  --port 30000

# language-only server
sglang serve \
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
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --enable-mm-global-cache \
  --port 30000

# language-only server
sglang serve \
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
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# language-only server
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

- EPD Disaggregation

```bash
# encoder 0
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# prefill 0
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
# decode 0
sglang serve \
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

When a language-only server starts, it automatically embeds bootstrap endpoints (`/register_encoder_url`, `/unregister_encoder_url`, `/list_encoder_urls`) in its HTTP server. Encoder servers use `--encoder-register-urls` pointing to one or more prefill servers to register on startup.

```bash
# Step 1: Start the language-only prefill server (bootstrap is embedded automatically)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# Step 2: Start encoders — they self-register with the prefill server
# Encoder 0
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://127.0.0.1:30002 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Encoder 1 (can be added later without restarting the prefill server)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://127.0.0.1:30002 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
```

You can also query the registered encoder URLs at any time:

```bash
curl http://127.0.0.1:30002/list_encoder_urls
# {"encoder_urls": ["http://127.0.0.1:30000", "http://127.0.0.1:30001"]}
```

### nEmP: Multiple Encoders, Multiple Prefill Servers (Per-Request Bootstrap)

In the nEmP scenario, you have **multiple encoder groups** and **multiple prefill servers**. Each incoming request specifies which prefill server to query for encoder discovery via the `epd_bootstrap_addr` field, allowing different requests on the same prefill server to use different sets of encoders.

Each prefill server embeds bootstrap endpoints automatically when `--language-only` is set. Encoders can register with **multiple** prefill servers via `--encoder-register-urls`, and requests carry `epd_bootstrap_addr` to select which prefill server's encoder group to use.

**Architecture:**

```
                      ┌──────────────────────────────────┐
 Request A ──────────►│  Prefill 0 (port 30002)          │──► Encoders {E0, E1}
 (epd_bootstrap_addr  └──────────────────────────────────┘
  = <PREFILL_0>:30002)
                      ┌──────────────────────────────────┐
 Request B ──────────►│  Prefill 1 (port 30003)          │──► Encoders {E2, E3}
 (epd_bootstrap_addr  └──────────────────────────────────┘
  = <PREFILL_1>:30003)
```

**Step 1: Start prefill servers (bootstrap is embedded):**

```bash
# Replace with actual host IPs/names in multi-host deployments;
# use 127.0.0.1 when testing on a single machine.
PREFILL_0_HOST=127.0.0.1
PREFILL_1_HOST=127.0.0.1   # use the second host's IP in production

# Prefill 0
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# Prefill 1
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30003
```

**Step 2: Start encoders and register with their prefill servers:**

An encoder can register with a single prefill server, or with **multiple** prefill servers at once using `--encoder-register-urls`:

```bash
# Encoder E0 (group A — registers with Prefill 0 only)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://${PREFILL_0_HOST}:30002 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Encoder E1 (registers with BOTH Prefill 0 and Prefill 1)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://${PREFILL_0_HOST}:30002 http://${PREFILL_1_HOST}:30003 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001

# Encoder E2 (group B — registers with Prefill 1 only)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://${PREFILL_1_HOST}:30003 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30004

# Encoder E3 (group B — also registers with Prefill 1)
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-register-urls http://${PREFILL_1_HOST}:30003 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30005
```

**Step 3: Send requests with `epd_bootstrap_addr`:**

```bash
# Request to Prefill 0, using encoder group A
curl http://${PREFILL_0_HOST}:30002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
      {"type": "text", "text": "Describe this image"}
    ]}],
    "epd_bootstrap_addr": "http://'"${PREFILL_0_HOST}"':30002"
  }'

# Request to Prefill 1, using encoder group B
curl http://${PREFILL_1_HOST}:30003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
      {"type": "text", "text": "Describe this image"}
    ]}],
    "epd_bootstrap_addr": "http://'"${PREFILL_1_HOST}"':30003"
  }'
```

> **Note:** The `epd_bootstrap_addr` field can also be used with the `/generate` endpoint. When a request carries `epd_bootstrap_addr`, it overrides the server-level encoder bootstrap for that request only.
>

#### gRPC Encoder (EPD)

You can run the encoder as a gRPC server while keeping prefill/decode as HTTP.
When using gRPC encoders, set `SGLANG_ENCODER_MM_RECEIVER_MODE=grpc` for the
prefill process so it uses the gRPC receiver.

```bash
# gRPC encoder
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --grpc-mode \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# prefill (HTTP) - tell it to use gRPC receiver
SGLANG_ENCODER_MM_RECEIVER_MODE=grpc \
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls grpc://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# decode (HTTP)
sglang serve \
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
