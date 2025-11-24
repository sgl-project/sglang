# Install SGLang

You can install SGLang using one of the methods below.

This page primarily applies to common NVIDIA GPU platforms.
For other or newer platforms, please refer to the dedicated pages for [AMD GPUs](../platforms/amd_gpu.md), [Intel Xeon CPUs](../platforms/cpu_server.md), [TPU](../platforms/tpu.md), [NVIDIA DGX Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/), [NVIDIA Jetson](../platforms/nvidia_jetson.md), [Ascend NPUs](../platforms/ascend_npu.md), and [Intel XPU](../platforms/xpu.md).

## Method 1: With pip or uv

It is recommended to use uv for faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang" --prerelease=allow
```

**Quick fixes to common problems**

- If you encounter `OSError: CUDA_HOME environment variable is not set`. Please set it to your CUDA install root with either of the following solutions:
  1. Use `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` to set the `CUDA_HOME` environment variable.
  2. Install FlashInfer first following [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html), then install SGLang as described above.

## Method 2: From source

```bash
# Use the last release branch
git clone -b v0.5.5.post3 https://github.com/sgl-project/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

**Quick fixes to common problems**

- If you want to develop SGLang, you can try the dev docker image. Please refer to [setup docker container](../developer_guide/development_guide_using_docker.md#setup-docker-container). The docker image is `lmsysorg/sglang:dev`.

## Method 3: Using docker

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).
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

You can also find the nightly docker images [here](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly).

## Method 4: Using Kubernetes

Please check out [OME](https://github.com/sgl-project/ome), a Kubernetes operator for enterprise-grade management and serving of large language models (LLMs).

<details>
<summary>More</summary>

1. Option 1: For single node serving (typically when the model size fits into GPUs on one node)

   Execute command `kubectl apply -f docker/k8s-sglang-service.yaml`, to create k8s deployment and service, with llama-31-8b as example.

2. Option 2: For multi-node serving (usually when a large model requires more than one GPU node, such as `DeepSeek-R1`)

   Modify the LLM model path and arguments as necessary, then execute command `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`, to create two nodes k8s statefulset and serving service.

</details>

## Method 5: Using docker compose

<details>
<summary>More</summary>

> This method is recommended if you plan to serve it as a service.
> A better approach is to use the [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.
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

## Method 7: Run on AWS SageMaker

<details>
<summary>More</summary>

To deploy on SGLang on AWS SageMaker, check out [AWS SageMaker Inference](https://aws.amazon.com/sagemaker/ai/deploy)

To host a model with your own container, follow the following steps:

1. Build a docker container with [sagemaker.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/sagemaker.Dockerfile) alongside the [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) script.
2. Push your container onto AWS ECR.

<details>
<summary>Dockerfile Build Script: <code>build-and-push.sh</code></summary>

```bash
#!/bin/bash
AWS_ACCOUNT="<YOUR_AWS_ACCOUNT>"
AWS_REGION="<YOUR_AWS_REGION>"
REPOSITORY_NAME="<YOUR_REPOSITORY_NAME>"
IMAGE_TAG="<YOUR_IMAGE_TAG>"

ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Starting build and push process..."

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Build the image
echo "Building Docker image..."
docker build -t ${IMAGE_URI} -f sagemaker.Dockerfile .

echo "Pushing ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Build and push completed successfully!"
```

</details>

3. Deploy a model for serving on AWS Sagemaker, refer to [deploy_and_serve_endpoint.py](https://github.com/sgl-project/sglang/blob/main/examples/sagemaker/deploy_and_serve_endpoint.py). For more information, check out [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk).
    1. By default, the model server on SageMaker will run with the following command: `python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080`. This is optimal for hosting your own model with SageMaker.
    2. To modify your model serving parameters, the [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) script allows for all available options within `python3 -m sglang.launch_server --help` cli by specifying environment variables with prefix `SM_SGLANG_`.
    3. The serve script will automatically convert all environment variables with prefix `SM_SGLANG_` from `SM_SGLANG_INPUT_ARGUMENT` into `--input-argument` to be parsed into `python3 -m sglang.launch_server` cli.
    4. For example, to run [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with reasoning parser, simply add additional environment variables `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` and `SM_SGLANG_REASONING_PARSER=qwen3`.

</details>

## Common Notes

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
- To reinstall flashinfer locally, use the following command: `pip3 install --upgrade flashinfer-python --force-reinstall --no-deps` and then delete the cache with `rm -rf ~/.cache/flashinfer`.
