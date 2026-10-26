---
title: "Disaggregated Diffusion Pipeline"
metatags:
    description: "Split SGLang Diffusion pipelines into independent encoder, denoiser, and decoder services for disaggregated serving."
---

Split a monolithic text-to-video/image pipeline into independent **Encoder**, **Denoiser**, and **Decoder** roles, each running on its own GPU(s). A central **DiffusionServer** routes requests through the pipeline.

## Quick Start

Disaggregation is controlled by a single flag: `--disagg-role`. Each component is launched independently, just like LLM PD disaggregation.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr>
      <th><code>--disagg-role</code></th>
      <th>What it runs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>monolithic</code></td>
      <td>(Default) Standard single-server mode</td>
    </tr>
    <tr>
      <td><code>encoder</code></td>
      <td>All stages with the default <code>RoleType.ENCODER</code> affinity: <code>InputValidationStage</code>, <code>TextEncodingStage</code> (plus <code>ImageEncodingStage</code> / <code>ImageVAEEncodingStage</code> for image-conditioned pipelines), <code>LatentPreparationStage</code>, <code>TimestepPreparationStage</code>, and any model-specific "before denoising" stage (e.g. <code>QwenImageLayeredBeforeDenoisingStage</code>, <code>GlmImageBeforeDenoisingStage</code>).</td>
    </tr>
    <tr>
      <td><code>prompt_encoder</code> / <code>image_encoder</code></td>
      <td>Separate text and reference-image encoding stages (currently FLUX.3 Image).</td>
    </tr>
    <tr>
      <td><code>denoiser</code></td>
      <td><code>DenoisingStage</code> (and its subclasses: <code>CausalDMDDenoisingStage</code>, <code>DmdDenoisingStage</code>, <code>LTX2AVDenoisingStage</code>, <code>LTX2RefinementStage</code>, <code>Hunyuan3DShapeDenoisingStage</code>, ...) — the DiT forward loop plus the scheduler stepping it drives.</td>
    </tr>
    <tr>
      <td><code>decoder</code></td>
      <td><code>DecodingStage</code> (VAE decode) and its subclasses (<code>LTX2AVDecodingStage</code>, <code>HeliosDecodingStage</code>, ...).</td>
    </tr>
    <tr>
      <td><code>server</code></td>
      <td>DiffusionServer head node + HTTP server (no GPU)</td>
    </tr>
  </tbody>
</table>

> Each stage declares its role via the `role_affinity` property on `PipelineStage` (default `ENCODER`). Non-monolithic pipelines only instantiate matching stages. The combined `encoder` role also includes stages with `PROMPT_ENCODER` or `IMAGE_ENCODER` affinity.

### Single-Machine Example (Verified)

The following commands have been tested end-to-end on an 8×H200 machine with
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. Each role runs on a separate GPU via
`--base-gpu-id`; the `server` head node requires no GPU.

```bash
# Terminal 1: Encoder (GPU 0)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 --base-gpu-id 0

# Terminal 2: Denoiser (GPU 1)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role denoiser \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19001 \
    --num-gpus 1 --base-gpu-id 1

# Terminal 3: Decoder (GPU 2)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 --base-gpu-id 2

# Terminal 4: DiffusionServer head (no GPU, receives HTTP requests)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://127.0.0.1:19000" \
    --denoiser-urls "tcp://127.0.0.1:19001" \
    --decoder-urls  "tcp://127.0.0.1:19002" \
    --host 0.0.0.0 --port 22000 \
    --scheduler-port 19655

# Send request (video generation)
curl http://127.0.0.1:22000/v1/videos \
    -H "Content-Type: application/json" \
    -d '{"model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "prompt": "A curious raccoon exploring a garden, cinematic", "size": "832x480"}'
```

> **Tested result (8×H200):**
> Encoder 2.3 s (TextEncoding) → Denoiser 312.8 s (50 steps, layerwise offload) → Decoder 7.1 s (VAE decode).
> Total ~322 s for 81-frame 1024×1024 video.

> **Tip:** `--base-gpu-id` controls which physical GPU the role uses.
> Encoder and Decoder can share a GPU (e.g. both `--base-gpu-id 0`) to save resources,
> but make sure the combined GPU memory is sufficient.

### Multi-Machine Example

The exact same CLI pattern — just replace `127.0.0.1` with actual IPs and add
RDMA flags for direct transfer:

```bash
# Machine A (10.0.0.1): Encoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 \
    --disagg-p2p-hostname 10.0.0.1 --disagg-ib-device mlx5_0

# Machine B (10.0.0.2): Denoiser (4 GPUs with SP)
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role denoiser \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19001 \
    --num-gpus 4 --denoiser-sp 4 --denoiser-ulysses 2 --denoiser-ring 2 \
    --disagg-p2p-hostname 10.0.0.2 --disagg-ib-device mlx5_0

# Machine C (10.0.0.3): Decoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 \
    --disagg-p2p-hostname 10.0.0.3 --disagg-ib-device mlx5_0

# Machine D (10.0.0.4): DiffusionServer head
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:19000" \
    --denoiser-urls "tcp://10.0.0.2:19001" \
    --decoder-urls  "tcp://10.0.0.3:19002" \
    --host 0.0.0.0 --port 30000 \
    --scheduler-port 19655 \
    --disagg-dispatch-policy max_free_slots
```

> ZMQ handles startup order gracefully — instances and head can start in any order.

## FLUX.3 split encoding {#flux3-split-encoding}

FLUX.3 Image supports both the three-role topology above and four independently
deployable worker pools:

```text
T2I: HTTP head -> prompt_encoder -> denoiser -> decoder -> HTTP head
i2i: HTTP head -> prompt_encoder -> image_encoder -> denoiser -> decoder -> HTTP head
```

The prompt encoder loads only the text encoder. The image encoder loads only
the VAE for reference-image encoding; the decoder has its own VAE instance.
The denoiser loads only the DiT and creates noise from the request seed locally.
T2I bypasses the image encoder and goes directly from prompt encoder to denoiser.
Stages within one request run sequentially; different requests can occupy
different worker pools concurrently. This is not a latency optimization for a
single request: separation adds transfers and allows independent placement and
scaling. The head still serves complete image-generation requests; this is not
a standalone API for returning training embeddings or VAE latents.

Use the same FLUX.3 package, variant, and numerical settings on every process.
Each host currently needs a complete package on disk, even though a worker
loads only its own components into memory. Follow the
[FLUX.3 cookbook](/cookbook/diffusion/FLUX/FLUX.3) for model access and licensing.
For gated checkpoints, authenticate on each host with an authorized account
using `hf auth login`, or provide an accessible local package path.
Install the transfer dependencies described below on every worker.
For long prompts, increase `--disagg-transfer-pool-size` on the encoding and
denoising workers to cover their largest incoming/outgoing tensors, as well as
reserved receive slots; for example, `1073741824` reserves a 1 GiB pool.

For a Linux NVIDIA deployment with four GPUs, start the workers in one shell:

```bash
MODEL_PATH=black-forest-labs/flux-3-image-klein
for spec in 'prompt_encoder 0 19000' 'image_encoder 1 19010' \
            'denoiser 2 19020' 'decoder 3 19030'; do
  read -r ROLE GPU WORK_PORT <<< "$spec"
  CUDA_VISIBLE_DEVICES="$GPU" sglang serve --model-path "$MODEL_PATH" \
    --num-gpus 1 --disagg-role "$ROLE" \
    --disagg-server-addr tcp://127.0.0.1:18000 \
    --disagg-p2p-hostname 127.0.0.1 \
    --scheduler-port "$WORK_PORT" --master-port "$((WORK_PORT + 50))" &
done
```

Then start the GPU-less head in another shell:

```bash
sglang serve --model-path black-forest-labs/flux-3-image-klein \
  --disagg-role server --host 127.0.0.1 --port 30000 --scheduler-port 18000 \
  --prompt-encoder-urls tcp://127.0.0.1:19000 \
  --image-encoder-urls tcp://127.0.0.1:19010 \
  --denoiser-urls tcp://127.0.0.1:19020 \
  --decoder-urls tcp://127.0.0.1:19030
```

The existing image API is unchanged: use `/v1/images/generations` for T2I and
multipart `/v1/images/edits` for references:

```bash
curl --fail-with-body -sS http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A sunset over mountains","size":"1024x1024","seed":42,"response_format":"b64_json"}' \
  -o flux3-t2i.json

curl --fail-with-body -sS http://127.0.0.1:30000/v1/images/edits \
  -F 'image=@reference.png' \
  -F 'prompt=Turn the scene into a watercolor painting' \
  -F 'seed=42' -F 'response_format=b64_json' \
  -o flux3-i2i.json
```

Replace `reference.png` with your input image. Omit `size` on edits to keep the
reference dimensions, including the model's padding/crop behavior. The head
inlines locally uploaded reference files before dispatch, so GPU workers do
not need access to the head's upload directory. URL inputs must remain
reachable from the image encoder.

Specify both `--prompt-encoder-urls` and `--image-encoder-urls`, instead of
`--encoder-urls`. Each accepts semicolon-separated worker endpoints, with the
same dispatch policy as the existing pools. Alternatively, use the existing
`encoder` role and `--encoder-urls` to colocate both encoders in one worker.
Image encoder and decoder processes can also share a GPU if their combined
weights, transfer pools, and concurrent activations fit.

Across machines, replace loopback addresses with routable IPs. Advertise each
worker's IP with `--disagg-p2p-hostname`, and use that exact work endpoint in
the head's URL list. The head also listens on scheduler port +4 for prompt
encoder results and +5 for image encoder results; +1/+2/+3 retain their
existing meanings. Keep worker work/control ports (+1), internal scheduler
ports, and distributed master ports distinct. Run the internal transport on a
trusted private network, not a public interface.

The BF16 flow-map package has been validated on H200 with same-node,
multi-process Mooncake RDMA: concurrent T2I, reference-size-preserving i2i,
and both three-role and four-role topologies matched the monolithic API's
output pixels with `quality=exact`. Cross-node execution has not been validated.

## Multiple Instances per Role

Use semicolons in `--*-urls` to register multiple instances:

```bash
# 2 encoders + 2 denoisers (4-GPU SP each) + 1 decoder
sglang serve --model-path ... --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:35000;tcp://10.0.0.2:35000" \
    --denoiser-urls "tcp://10.0.0.3:35000;tcp://10.0.0.4:35000" \
    --decoder-urls  "tcp://10.0.0.5:35000"
```

### GLM-Image distributed mode

GLM-Image can batch AR generation in the head and dispatch the resulting prior
tokens to distributed denoiser workers. Each worker runs prompt/glyph preparation,
DiT, and VAE decoding locally; no latent or embedding tensors are transferred.

The following 16-device deployment uses devices 0-1 for the external AR server
and devices 2-15 for 14 independent batch-1 Cache-DiT denoisers.

```bash
# Run 14 distributed denoisers
DISAGG_SERVER="tcp://127.0.0.1:19655"
MODEL_PATH="zai-org/GLM-Image"
BASE_MASTER_PORT=29005

export SGLANG_CACHE_DIT_FN=2
export SGLANG_CACHE_DIT_BN=1
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.4
export SGLANG_CACHE_DIT_MC=4
export SGLANG_CACHE_DIT_TAYLORSEER=true
export SGLANG_CACHE_DIT_TS_ORDER=2
export SGLANG_CACHE_DIT_ENABLED=true

worker_pids=()

cleanup() {
    trap - EXIT
    if ((${#worker_pids[@]})); then
        kill "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for i in $(seq 2 15); do
    scheduler_port=$((19001 + i))
    master_port=$((BASE_MASTER_PORT + i))

    sglang serve \
        --model-path "$MODEL_PATH" \
        --disagg-role denoiser \
        --disagg-server-addr "$DISAGG_SERVER" \
        --srt-encoder-url http://127.0.0.1:30020 \
        --scheduler-port "$scheduler_port" \
        --master-port "$master_port" \
        --num-gpus 1 \
        --base-gpu-id "$i" \
        --denoiser-sp 1 \
        --cfg-parallel-size 1 \
        --batching-max-size 1 \
        --dit-cpu-offload false \
        --attention-backend fa &
    worker_pids+=("$!")
done

# Stop all denoisers if any worker exits or fails during startup.
wait -n "${worker_pids[@]}"
```

Run the external AR server:

```bash
sglang serve \
  --model-path ./zai-org/GLM-Image/vision_language_encoder/ \
  --tokenizer-path ./zai-org/GLM-Image/processor/ \
  --enable-multimodal \
  --cuda-graph-max-bs-decode 28 \
  --device npu \
  --attention-backend ascend \
  --disable-fast-image-processor \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30020 \
  --mem-fraction-static 0.8
```

Run the public head. `--encoder-urls` and `--decoder-urls` are intentionally
omitted for this topology.

```bash
sglang serve \
  --model-path zai-org/GLM-Image \
  --disagg-role server \
  --srt-encoder-url http://127.0.0.1:30020 \
  --srt-encoder-timeout 300 \
  --denoiser-urls "tcp://127.0.0.1:19003;tcp://127.0.0.1:19004;tcp://127.0.0.1:19005;tcp://127.0.0.1:19006;tcp://127.0.0.1:19007;tcp://127.0.0.1:19008;tcp://127.0.0.1:19009;tcp://127.0.0.1:19010;tcp://127.0.0.1:19011;tcp://127.0.0.1:19012;tcp://127.0.0.1:19013;tcp://127.0.0.1:19014;tcp://127.0.0.1:19015;tcp://127.0.0.1:19016" \
  --batching-mode dynamic \
  --batching-max-size 28 \
  --batching-delay-ms 30 \
  --enable-batching-metrics \
  --host 0.0.0.0 \
  --port 30052 \
  --scheduler-port 19655 \
  --output-path ./outputs
```

Workers return decoded pixels to the head, which saves and serves the final
files from `--output-path`. The PR benchmark used
[longtext-bench.zip](https://github.com/user-attachments/files/29779516/longtext-bench.zip):

```bash
python fetch_images.py \
  --base-url http://localhost:30052/v1 \
  --model GLM-Image-distributed-test \
  --output-dir generated_images \
  --max-concurrency 56
```

## Port Convention

Result endpoints are derived deterministically from the head node's `--scheduler-port` (default: 5555):

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Socket</th>
      <th>Port</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DS frontend (ROUTER)</td>
      <td><code>scheduler_port</code></td>
    </tr>
    <tr>
      <td>Encoder result (PULL)</td>
      <td><code>scheduler_port + 1</code></td>
    </tr>
    <tr>
      <td>Denoiser result (PULL)</td>
      <td><code>scheduler_port + 2</code></td>
    </tr>
    <tr>
      <td>Decoder result (PULL)</td>
      <td><code>scheduler_port + 3</code></td>
    </tr>
    <tr>
      <td>Prompt encoder result (PULL, split encoding)</td>
      <td><code>scheduler_port + 4</code></td>
    </tr>
    <tr>
      <td>Image encoder result (PULL, split encoding)</td>
      <td><code>scheduler_port + 5</code></td>
    </tr>
  </tbody>
</table>

Role instances derive their result endpoint automatically from `--disagg-server-addr`. No manual endpoint configuration needed.

## Transfer Mechanism

Tensor data between adjacent roles is transferred via a P2P transfer engine. The DiffusionServer routes control messages and request metadata; tensors flow directly between workers. Uploaded reference files are forwarded inline as request metadata.

**mooncake-transfer-engine** is required for disaggregated diffusion. It provides RDMA for direct GPU-to-GPU data movement.

The GLM-Image distributed mode is an exception: it relays only prior token IDs
and request metadata over ZMQ and does not require Mooncake.

```bash
pip install mooncake-transfer-engine
```

### Transfer Flow

1. **Sender** stages output tensors in its transfer buffer, then notifies DiffusionServer.
2. **DiffusionServer** selects a downstream worker and reserves one of its preallocated receive slots. If none fits, it sends `transfer_alloc` and waits for `transfer_allocated`.
3. **DiffusionServer** sends `transfer_push` to the sender with the receiver's session and destination address.
4. **Sender** pushes the tensors through Mooncake and reports `transfer_pushed` after completion.
5. **DiffusionServer** sends `transfer_ready` to the receiver and releases the sender's scheduling slot.
6. **Receiver** loads the tensors on its transfer stream and executes its stages. Receive prefetch can overlap the previous request's compute.

Decoder results (final output) flow back through DiffusionServer as raw ZMQ frames to the HTTP client.

### Failure Isolation

The head rejects malformed scheduler requests and duplicate active request IDs
without replacing the original request. Each scheduler message carries one
request; use concurrent API calls for independent requests, or `n` for multiple
outputs from one request.

Tensor decoding errors fail only the associated request when its ID can be
recovered. Results without an identifiable request are logged and discarded;
affected requests still rely on `--disagg-timeout`. Reported worker errors release
their scheduling capacity and reserved receive slots so later requests can run.

Keep scheduler and worker ports on a trusted private network. This error handling
does not make the pickle-based scheduler protocol safe for untrusted clients, or
provide durable jobs and transparent recovery after a head restart.

### RDMA Flags

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Flag</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--disagg-p2p-hostname</code></td>
      <td><code>127.0.0.1</code></td>
      <td>RDMA-reachable hostname/IP of this instance</td>
    </tr>
    <tr>
      <td><code>--disagg-ib-device</code></td>
      <td><code>None</code></td>
      <td>InfiniBand device (e.g., <code>mlx5_0</code>, <code>mlx5_roce0</code>)</td>
    </tr>
    <tr>
      <td><code>--disagg-transfer-pool-size</code></td>
      <td>256 MiB</td>
      <td>Pinned memory pool per instance</td>
    </tr>
  </tbody>
</table>

Set `--disagg-p2p-hostname` to the actual IP on each machine. For multi-machine, `--disagg-ib-device` specifies the RDMA NIC.

## Per-Role Parallelism

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Flag</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--encoder-tp</code></td>
      <td>Encoder tensor parallelism</td>
    </tr>
    <tr>
      <td><code>--denoiser-tp</code> / <code>--denoiser-sp</code> / <code>--denoiser-ulysses</code> / <code>--denoiser-ring</code></td>
      <td>Denoiser parallelism</td>
    </tr>
    <tr>
      <td><code>--decoder-sp</code></td>
      <td>Decoder sequence parallelism</td>
    </tr>
  </tbody>
</table>

If not specified, parallelism is auto-derived from `--num-gpus`.

## Other Options

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Flag</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--disagg-timeout</code></td>
      <td><code>600</code></td>
      <td>Timeout (seconds) for pending requests</td>
    </tr>
    <tr>
      <td><code>--disagg-dispatch-policy</code></td>
      <td><code>round_robin</code></td>
      <td><code>round_robin</code> or <code>max_free_slots</code></td>
    </tr>
  </tbody>
</table>

## Python API

For programmatic single-machine deployment, `launch_pool_disagg_server()` is available. It spawns its workers, and each child re-executes this script's module scope before initializing its platform, so keep the diffusion imports inside the guard:

```python
if __name__ == "__main__":
    from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    server_args = ServerArgs.from_kwargs(
        model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        denoiser_sp=4, denoiser_ulysses=2, denoiser_ring=2,
        disagg_ib_device="mlx5_0",
    )

    launch_pool_disagg_server(
        server_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1, 2, 3, 4], [5, 6, 7, 8]],
        decoder_gpus=[[0]],
    )
```

## Architecture

```
Client ─── HTTP (port 30000) ──► FastAPI Server
                                      │
                                      ▼
                              DiffusionServer (ROUTER, scheduler_port)
                              ┌───────┼───────┐
                   PUSH work  │       │       │  PUSH work
                              ▼       │       ▼
                    Encoder[0..N]     │    Decoder[0..K]
                              │       │       ▲
                   P2P tensor │       │       │ P2P tensor
                   transfer   ▼       │       │ transfer
                          Denoiser[0..M] ─────┘
                                      │
                    PULL results ◄────┘  (decoder → DS → client)
```

### Request State Machine

```
PENDING → ENCODER_WAITING → ENCODER_RUNNING → ENCODER_DONE
                                                    │
                        DENOISING_WAITING → DENOISING_RUNNING → DENOISING_DONE
                                                                       │
                                    DECODER_WAITING → DECODER_RUNNING → DONE
```

Any state can transition to `FAILED` or `TIMED_OUT`.
