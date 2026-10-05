---
title: Installation
description: Install SGLang with pip/uv, source, Docker, Kubernetes, and cloud deployment options.
keywords:
  - installation
  - sglang
  - pip
  - docker
---
Choose one installation method below. These instructions target Linux with NVIDIA GPUs; for other devices, see [Hardware platforms](/docs/hardware-platforms/overview).

## Install SGLang

<Tabs>
<Tab title="Docker" id="method-3-using-docker">

Docker includes Python, SGLang, and its dependencies; no separate host Python installation is required.

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).
Use Linux with a supported NVIDIA GPU, a CUDA 13-compatible driver, [Docker Engine](https://docs.docker.com/engine/install/), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Complete the Toolkit guide's Docker runtime configuration before starting a GPU container.

SGLang images ship a CUDA 13 environment. CUDA 12 images (`-cu12` / `-cu129`) are no longer published; `lmsysorg/sglang:v0.5.19-cu129` is the last CUDA 12 tag.

**Pull the image**

`latest` contains the newest stable release:


```bash
docker pull lmsysorg/sglang:latest
```

**Image options**

The examples use `latest`. To use another image, replace the tag in the pull and run commands:

- Nightly: Built daily from the latest `main`, including changes not yet in a stable release. Choose a tag from the [nightly image list](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly).
- Runtime: A smaller image of the same stable release, with inference dependencies and fewer development utilities. Use the `latest-runtime` tag.

For reproducible deployments, use a specific release or dated nightly tag from [Docker Hub](https://hub.docker.com/r/lmsysorg/sglang/tags).

**Start a container**

Set `HF_CACHE_DIR` to an absolute directory path on the host for downloaded model weights, replacing the placeholder below.

For gated or private models, export your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) as `HF_TOKEN` on the host and add `--env HF_TOKEN` before the image name in this command.


```bash
HF_CACHE_DIR="/path/to/huggingface-cache"
docker run -it --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  --ipc=host \
  lmsysorg/sglang:latest bash
```

The container shell is ready for `sglang serve`.

</Tab>
<Tab title="pip / uv" id="method-1-with-pip-or-uv">

Use Python 3.10 or higher and a compatible CUDA 13 environment.

<Accordion title="CUDA version compatibility">

SGLang requires CUDA 13. CUDA 12 (`cu129`) wheels and images were retired with the upgrade to PyTorch 2.14, which publishes no CUDA 12.9 builds. SGLang 0.5.19 is the last release with a CUDA 12 lane (PyTorch 2.13).

</Accordion>

**Prepare the environment**

Use uv for faster installation. Install it with pip, or follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

```bash
pip install --upgrade pip
pip install uv
```

Create and activate a virtual environment:

```bash Command
uv venv --python 3.12
source .venv/bin/activate
```

**Install SGLang**

```bash
uv pip install --prerelease=allow sglang
```

<Accordion title="Why --prerelease=allow?">

Some of SGLang's dependencies only publish pre-releases on PyPI, so without `--prerelease=allow` uv older than 0.12.0 silently installs SGLang 0.5.9. On [uv 0.12.0](https://github.com/astral-sh/uv/releases/tag/0.12.0) and newer the flag is a harmless no-op.
</Accordion>

<Accordion title="Nightly builds" id="nightly-builds">

To pick up the latest features and fixes before the next stable release, install a nightly build. Nightly wheels are built from the latest `main` and published to the SGLang wheel index. Add that index with `--extra-index-url`, and combine `--prerelease=allow` with `--index-strategy unsafe-best-match` so uv considers the nightly (pre-release) version alongside PyPI:

```bash Command
uv pip install --prerelease=allow --index-strategy unsafe-best-match --extra-index-url https://docs.sglang.ai/whl/cu130/ sglang
```

</Accordion>

For `CUDA_HOME` errors, see [Troubleshooting](#cuda_home).

</Tab>
<Tab title="From source" id="method-2-from-source">

Use Python 3.10 or higher in an activated Python environment, with CUDA dependencies appropriate for the version you build.

**Get the source**

Clone a release tag (the example below pins one); choose another from [GitHub releases](https://github.com/sgl-project/sglang/releases) as needed:

```bash
git clone -b v0.5.21 https://github.com/sgl-project/sglang.git
cd sglang
```

**Install SGLang**

Install the Python package in editable mode so local source changes are reflected in your environment:

```bash
pip install --upgrade pip
pip install -e "python"
```

<Accordion title="Development environment">

For contributing to SGLang, follow the [development environment guide](/docs/developer_guide/development_guide_using_docker#setup-docker-container), which covers a container with development tools and dependencies installed.

</Accordion>

For `CUDA_HOME` errors, see [Troubleshooting](#cuda_home).

</Tab>
</Tabs>

## Start a model server

Start a model server inside your Docker container or activated Python environment. Replace `MODEL_PATH` with a Hugging Face model ID or a local model directory:

```bash
sglang serve MODEL_PATH --host 0.0.0.0 --port 30000
```

Additional launch arguments depend on the model and your hardware.

- [Quickstart](/docs/get-started/quickstart): Launch a model and send a request to verify the server.
- [Cookbook](/cookbook/intro): Find model-specific launch arguments for your hardware.

## Deployment options

Choose how to run SGLang as a service. Expand an option for setup instructions.

<AccordionGroup>

<a id="method-4-using-kubernetes" />

<Accordion title="Kubernetes" id="kubernetes" description="Serve models on a Kubernetes cluster, on one node or across multiple nodes.">

Please check out [OME](https://github.com/sgl-project/ome), a Kubernetes operator for enterprise-grade management and serving of large language models (LLMs).

1. Option 1: For single node serving (typically when the model size fits into GPUs on one node)

   Execute command `kubectl apply -f docker/k8s-sglang-service.yaml`, to create k8s deployment and service, with llama-31-8b as example.

2. Option 2: For multi-node serving (usually when a large model requires more than one GPU node, such as `DeepSeek-R1`)

   Modify the LLM model path and arguments as necessary, then execute command `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`, to create two nodes k8s statefulset and serving service.

</Accordion>

<a id="method-5-using-docker-compose" />

<Accordion title="Docker Compose" id="docker-compose" description="Run a model service on a single Docker host with a Compose configuration.">

> This method is recommended if you plan to serve it as a service.
> A better approach is to use the [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml).

1. Copy the [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) to your local machine
2. Execute the command `docker compose up -d` in your terminal.

</Accordion>

<a id="method-6-run-on-kubernetes-or-clouds-with-skypilot" />

<Accordion title="SkyPilot" id="skypilot" description="Deploy on Kubernetes or across cloud providers from one YAML configuration.">

To deploy on Kubernetes or 12+ clouds, you can use [SkyPilot](https://github.com/skypilot-org/skypilot).

1. Install SkyPilot and set up Kubernetes cluster or cloud access: see [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
2. Deploy on your own infra with a single command and get the HTTP API endpoint:

<Accordion title="SkyPilot YAML: sglang.yaml">

```yaml Config
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

</Accordion>

```bash Command
# Deploy on any cloud or Kubernetes cluster. Use --cloud <cloud> to select a specific cloud provider.
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang
```

3. To further scale up your deployment with autoscaling and failure recovery, check out the [SkyServe + SGLang guide](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve).

</Accordion>

<a id="method-7-run-on-aws-sagemaker" />

<Accordion title="AWS SageMaker" id="aws-sagemaker" description="Deploy on SageMaker with a pre-built SGLang image or your own container.">

To deploy on SGLang on AWS SageMaker, check out [AWS SageMaker Inference](https://aws.amazon.com/sagemaker/ai/deploy)

Amazon Web Services provide supports for SGLang containers along with routine security patching. For available SGLang containers, check out [AWS SGLang DLCs](https://aws.github.io/deep-learning-containers/reference/available_images/#sglang).

To deploy a pre-built SGLang Deep Learning Container without building your own image, see [Amazon SageMaker AI](/docs/basic_usage/aws_sagemaker).

To host a model with your own container, follow the following steps:

1. Build a docker container with [sagemaker.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/sagemaker.Dockerfile) alongside the [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) script.
2. Push your container onto AWS ECR.

<Accordion title="Build script: build-and-push.sh">

```bash Command
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

</Accordion>

3. Deploy a model for serving on AWS Sagemaker, refer to [deploy_and_serve_endpoint.py](https://github.com/sgl-project/sglang/blob/main/examples/sagemaker/deploy_and_serve_endpoint.py). For more information, check out [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk).
   1. By default, the model server on SageMaker will run with the following command: `python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080`. This is optimal for hosting your own model with SageMaker.
   2. To modify your model serving parameters, the [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) script allows for all available options within `python3 -m sglang.launch_server --help` cli by specifying environment variables with prefix `SM_SGLANG_`.
   3. The serve script will automatically convert all environment variables with prefix `SM_SGLANG_` from `SM_SGLANG_INPUT_ARGUMENT` into `--input-argument` to be parsed into `python3 -m sglang.launch_server` cli.
   4. For example, to run [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with reasoning parser, simply add additional environment variables `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` and `SM_SGLANG_REASONING_PARSER=qwen3`.

</Accordion>

</AccordionGroup>

<a id="common-notes" />

## Troubleshooting

### CUDA_HOME

If installation fails with `OSError: CUDA_HOME environment variable is not set`, set `CUDA_HOME` to your CUDA Toolkit installation directory. Replace the path below with the actual path on your machine:

```bash
export CUDA_HOME="/path/to/cuda"
```

For a versioned installation, this is typically `/usr/local/cuda-VERSION`. Alternatively, install FlashInfer first using the [FlashInfer installation instructions](https://docs.flashinfer.ai/installation.html), then retry the SGLang installation.

### FlashInfer

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
- To reinstall flashinfer locally, use the following command: `pip3 install --upgrade flashinfer-python --force-reinstall --no-deps` and then delete the cache with `rm -rf ~/.cache/flashinfer`.
