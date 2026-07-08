# MiMo-V2.5-Pro Usage on AMD Instinct MI355X

[MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) is Xiaomi's most capable reasoning model, with 1T total parameters. It is purpose-built for complex agentic tasks including large-scale code generation, deep data analysis, and long-horizon execution. The model supports a 1M-token context window and features Multi-Token Prediction for faster inference.

SGLang provides Day-0 support for MiMo-V2.5-Pro on AMD Instinct MI355X GPUs with SpecV2-based multi-layer EAGLE speculative decoding.

## Supported Models

- [XiaomiMiMo/MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)

## System Requirements

- **Hardware**: 8x AMD Instinct MI355X GPUs (288 GB on-chip memory, 8 TB/s bandwidth)
- **Software**: AMD ROCm 7, SGLang with EAGLE speculative decoding support

## Deployment with SGLang

### Step 1: Launch Docker Container

Use the pre-built upstream Docker image for MI355X:

```bash
docker run -d -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 32G \
    --entrypoint "/bin/bash" \
    --name mimov25pro \
    aigmkt/mimo-v2.5-pro-sglang:latest
```

### Step 2: Start SGLang Server

Inside the container, launch the server with EAGLE speculative decoding enabled:

```bash
export SGLANG_ENABLE_SPEC_V2=1

python3 -m sglang.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5-Pro \
  --tp-size 8 \
  --decode-log-interval 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --disable-radix-cache \
  --watchdog-timeout 1000000 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 131072 \
  --max-running-requests 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --attention-backend triton
```

**Key flags explained:**

- `SGLANG_ENABLE_SPEC_V2=1`: Enables the experimental overlap scheduler for EAGLE speculative decoding, improving performance by overlapping draft and verification stages.
- `--speculative-algorithm EAGLE --enable-multi-layer-eagle`: Activates multi-layer EAGLE speculative decoding for faster inference.
- `--attention-backend triton`: Uses the Triton attention backend, validated for MI355X.
- `--chunked-prefill-size 131072`: Sets a large chunked prefill size to handle long-context inputs efficiently.
- `--disable-radix-cache`: Disables radix cache for compatibility.

### Step 3: Test the Deployment

Once the server is running, send a chat completion request from another terminal:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "XiaomiMiMo/MiMo-V2.5-Pro",
        "messages": [
            {"role": "user", "content": "What is the history of Beijing?"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

## Resources

- [MiMo Open Platform](https://platform.xiaomimimo.com)
- [AMD ROCm AI Developer Hub](https://www.amd.com/en/developer/resources/rocm-hub.html)
- [AMD Instinct MI355X](https://www.amd.com/en/products/accelerators/instinct.html)
- Docker Image: `aigmkt/mimo-v2.5-pro-sglang:latest`
