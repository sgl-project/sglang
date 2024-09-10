# SGLang Installation Guide

SGLang consists of a frontend language (Structured Generation Language, SGLang) and a backend runtime (SGLang Runtime, SRT). The frontend can be used separately from the backend, allowing for a detached frontend-backend setup.

## Quick Installation Options

### 1. Frontend Installation (Client-side, any platform)

```bash
pip install --upgrade pip
pip install sglang
```

**Note: You can check [these examples](https://github.com/sgl-project/sglang/tree/main/examples/frontend_language/usage) for how to use frontend and backend separately.**

### 2. Backend Installation (Server-side, Linux only)

```bash
pip install --upgrade pip
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Note: The backend (SRT) is only needed on the server side and is only available for Linux right now.**

**Important: Please check the [flashinfer installation guidance](https://docs.flashinfer.ai/installation.html) to install the proper version according to your PyTorch and CUDA versions.**

### 3. From Source (Latest version, Linux only for full installation)

```bash
# Use the latest release branch
# As of this documentation, it's v0.2.15, but newer versions may be available
# Do not clone the main branch directly; always use a specific release version
# The main branch may contain unresolved bugs before a new release
git clone -b v0.2.15 https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### 4. OpenAI Backend Only (Client-side, any platform)

If you only need to use the OpenAI backend, you can avoid installing other dependencies by using:

```bash
pip install "sglang[openai]"
```

## Advanced Installation Options

### 1. Using Docker (Server-side, Linux only)

The docker images are available on Docker Hub as [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker). Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

### 2.Using docker compose

This method is recommended if you plan to serve it as a service. A better approach is to use the [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.

### 3.Run on Kubernetes or Clouds with SkyPilot

<details>
<summary>More</summary>

To deploy on Kubernetes or 12+ clouds, you can use [SkyPilot](https://github.com/skypilot-org/skypilot).

1. Install SkyPilot and set up Kubernetes cluster or cloud access: see [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
2. Deploy on your own infra with a single command and get the HTTP API endpoint:
<details>
<summary>SkyPilot YAML: <code>sglang.yaml</code></summary>

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```
</details>

```bash
# Deploy on any cloud or Kubernetes cluster. Use --cloud <cloud> to select a specific cloud provider.
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang
```
3. To further scale up your deployment with autoscaling and failure recovery, check out the [SkyServe + SGLang guide](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve).
</details>

## Troubleshooting

- For FlashInfer issues on newer GPUs, use `--disable-flashinfer --disable-flashinfer-sampling` when launching the server.
- For out-of-memory errors, try `--mem-fraction-static 0.7` when launching the server.

For more details and advanced usage, visit the [SGLang GitHub repository](https://github.com/sgl-project/sglang).