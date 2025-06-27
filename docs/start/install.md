# Install SGLang

You can install SGLang using any of the methods below.

For running DeepSeek V3/R1, refer to [DeepSeek V3 Support](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3). It is recommended to use the latest version and deploy it with [Docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended) to avoid environment-related issues.

It is recommended to use uv to install the dependencies for faster installation:

## Method 1: With pip or uv

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.8.post1"
```

**Quick Fixes to Common Problems**

- SGLang currently uses torch 2.7.1, so you need to install flashinfer for torch 2.7.1. If you want to install flashinfer separately, please refer to [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html). Please note that the FlashInfer pypi package is called `flashinfer-python` instead of `flashinfer`.

- If you encounter `OSError: CUDA_HOME environment variable is not set`. Please set it to your CUDA install root with either of the following solutions:

  1. Use `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` to set the `CUDA_HOME` environment variable.
  2. Install FlashInfer first following [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html), then install SGLang as described above.

## Method 2: From source

```bash
# Use the last release branch
git clone -b v0.4.8.post1 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

Note: SGLang currently uses torch 2.7.1, so you need to install flashinfer for torch 2.7.1. If you want to install flashinfer separately, please refer to [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html).

If you want to develop SGLang, it is recommended to use docker. Please refer to [setup docker container](https://github.com/sgl-project/sglang/blob/main/docs/references/development_guide_using_docker.md#setup-docker-container) for guidance. The docker image is `lmsysorg/sglang:dev`.

Note: For AMD ROCm system with Instinct/MI GPUs, do following instead:

```bash
# Use the last release branch
git clone -b v0.4.8.post1 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install
cd ..
pip install -e "python[all_hip]"
```

## Method 3: Using docker

The docker images are available on Docker Hub as [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).
Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

Note: For AMD ROCm system with Instinct/MI GPUs, it is recommended to use `docker/Dockerfile.rocm` to build images, example and usage as below:

```bash
docker build --build-arg SGL_BRANCH=v0.4.8.post1 -t v0.4.8.post1-rocm630 -f Dockerfile.rocm .

alias drun='docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host \
    --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx -v /data:/data'

drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    v0.4.8.post1-rocm630 \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000

# Till flashinfer backend available, --attention-backend triton --sampling-backend pytorch are set by default
drun v0.4.8.post1-rocm630 python3 -m sglang.bench_one_batch --batch-size 32 --input 1024 --output 128 --model amd/Meta-Llama-3.1-8B-Instruct-FP8-KV --tp 8 --quantization fp8
```

## Method 4: Using docker compose

<details>
<summary>More</summary>

> This method is recommended if you plan to serve it as a service.
> A better approach is to use the [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.
</details>

## Method 5: Using Kubernetes

<details>
<summary>More</summary>

1. Option 1: For single node serving (typically when the model size fits into GPUs on one node)

   Execute command `kubectl apply -f docker/k8s-sglang-service.yaml`, to create k8s deployment and service, with llama-31-8b as example.

2. Option 2: For multi-node serving (usually when a large model requires more than one GPU node, such as `DeepSeek-R1`)

   Modify the LLM model path and arguments as necessary, then execute command `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`, to create two nodes k8s statefulset and serving service.

</details>

## Method 6: Run on Kubernetes or Clouds with SkyPilot

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

## Common Notes

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
- If you only need to use OpenAI models with the frontend language, you can avoid installing other dependencies by using `pip install "sglang[openai]"`.
- The language frontend operates independently of the backend runtime. You can install the frontend locally without needing a GPU, while the backend can be set up on a GPU-enabled machine. To install the frontend, run `pip install sglang`, and for the backend, use `pip install sglang[srt]`. `srt` is the abbreviation of SGLang runtime.
- To reinstall flashinfer locally, use the following command: `pip3 install --upgrade flashinfer-python --force-reinstall --no-deps` and then delete the cache with `rm -rf ~/.cache/flashinfer`.
