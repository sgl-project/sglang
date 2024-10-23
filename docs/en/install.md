## Install SGLang

You can install SGLang using any of the methods below.

### Method 1: With pip
```
pip install --upgrade pip
pip install "sglang[all]"

# Install FlashInfer accelerated kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Important: Please check the [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html) to install the proper version according to your PyTorch and CUDA versions.**

### Method 2: From source
```
# Use the last release branch
git clone -b v0.3.4.post1 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"

# Install FlashInfer accelerated kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Important: Please check the [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html) to install the proper version according to your PyTorch and CUDA versions.**

### Method 3: Using docker
The docker images are available on Docker Hub as [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).
Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

### Method 4: Using docker compose

<details>
<summary>More</summary>

> This method is recommended if you plan to serve it as a service.
> A better approach is to use the [k8s-sglang-service.yaml](./docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](./docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.
</details>

### Method 5: Run on Kubernetes or Clouds with SkyPilot

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
    --model-path meta-llama/Llama-3.1-8B-Instruct \
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

### Common Notes
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
- If you only need to use the OpenAI backend, you can avoid installing other dependencies by using `pip install "sglang[openai]"`.
- The language frontend operates independently of the backend runtime. You can install the frontend locally without needing a GPU, while the backend can be set up on a GPU-enabled machine. To install the frontend, run `pip install sglang`, and for the backend, use `pip install sglang[srt]`. This allows you to build SGLang programs locally and execute them by connecting to the remote backend.
