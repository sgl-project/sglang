---
title: "DeepSeek-V4-Flash"
metatags:
  description: "Deploy DeepSeek-V4-Flash model with SGLang on A3 and 950PR/DT series NPUs, including multi-node PD disaggregation mode."
---

## Introduction

DeepSeek-V4-Flash is a Mixture-of-Experts (MoE) large language model developed by DeepSeek, featuring 284B total
parameters with 13B active parameters. It employs hybrid sparse attention (CSA + HCA) and DeepSeekMoE architecture with
a bundled multi-token prediction (MTP) draft head for speculative decoding, supporting a 1M-token context length.

This document demonstrates the deployment of DeepSeek-V4-Flash on Ascend NPUs using SGLang, covering both the A3
Series and the 950PR/DT Series (SoC name A5), including single-node PD mixed deployment, multi-node PD disaggregation
mode, feature configuration, and performance optimization.

This document is validated and written based on **SGLang v0.5.16**. The current model (DeepSeek-V4-Flash) is fully
supported in this version. To use the latest features (e.g., PD disaggregation, speculative decoding), it is
recommended to use v0.5.16 or a later version.

## Supported features

| Feature                       | Example usage                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| Attention Backend             | `--attention-backend dsv4`                                                                    |
| Tensor Parallelism            | `--tp-size 16`                                                                                |
| Data Parallelism              | `--dp-size 16`                                                                                |
| Expert Parallelism            | `--ep-size 16 \`<br/>`--moe-a2a-backend deepep \`<br/>`--deepep-mode auto`                    |
| PD Disaggregation             | `--disaggregation-mode prefill \`<br/>`--disaggregation-transfer-backend ascend`              |
| Quantization                  | `--quantization modelslim`                                                                    |
| NPU Graph                     | enabled by default; disable with `--disable-cuda-graph`;<br/>control range via `--cuda-graph-bs-decode` or `--cuda-graph-max-bs-decode`; e.g., `--cuda-graph-bs-decode 1 2 4 8 16 24 36 40 48 56` |
| Speculative Decoding          | `--speculative-algorithm EAGLE \`<br/>`--speculative-num-steps 2 \`<br/>`--speculative-eagle-topk 1 \`<br/>`--speculative-num-draft-tokens 3` |
| Overlap Schedule              | `export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`                                                  |
| DP LM Head                    | `--enable-dp-lm-head`                                                                         |
| DP Attention                  | `--dp-size 16 \`<br/>`--enable-dp-attention`                                                                           |

<Note>
The values in the **Example usage** column are for illustration only. Adjust them according to your hardware, deployment
mode, and workload. For parameter details, see
[Feature descriptions](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning#feature-descriptions); for
recommended configurations for each deployment scenario, see [Best practices](#best-practices).
</Note>

For feature compatibility and conflict information between features,
see [Feature Compatibility](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning#feature-compatibility).

## Prerequisites

### Environment

Before following this tutorial, complete the environment setup in the documents below:

- [Ascend NPU Quickstart](/docs/hardware-platforms/ascend-npus/getting-started/quick_start) — the fastest way to get started.
  It walks you through launching the official container image, starting the SGLang server, and sending a test request.
  Recommended if you are new to SGLang on Ascend.
- [SGLang Installation with NPU Support](/docs/hardware-platforms/ascend-npus/getting-started/installation) — the full installation guide.
  It covers the component version mapping (CANN, TorchNPU, Triton, kernels, etc.), building from source or from a
  Dockerfile, and recommended system settings (CPU power scheme, NUMA, swap). Use it when you need to install or customize
  the environment instead of using the official image.

### Model weights

Before downloading model weights, check the model size to reserve enough disk space.
For multi-node deployment, download the weights to a shared directory accessible to all nodes.

- [DeepSeek-V4-Flash-w8a8-mtp](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp) (W8A8 quantized version, 300GB)
- [DeepSeek-V4-Flash-0731](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731) (Native FP8 version, 155.44GB, for the 950PR/DT Series)

<Info>
The W8A8 variant (300GB) can be deployed on 8 × 64GB of device memory (`--tp-size 8`), which corresponds to
8 dies on an A3 Series node (4 cards). The native FP8 variant (155.44GB) runs with `--tp-size 4` on the
950PR/DT Series (4 cards).
</Info>

This is the minimum recommended configuration. For optimized configurations,
see [Best practices](#best-practices), which may require additional cards or nodes.

For the hardware specifications (memory per die and dies per card),
see [Ascend NPU Reference — Hardware](/docs/hardware-platforms/ascend-npus/reference/glossary#hardware).

## Installation

<Warning>
Ensure sufficient disk space before pulling images. The Docker image requires at least **30GB** of free space.
</Warning>

The dependencies required for the NPU runtime environment have been integrated into a Docker image and uploaded to the
online platform. You can directly pull it.

Both **stable releases** and **daily builds** are available. CANN 9.1.0 has no stable release yet, so the following
commands use the validated build tags `cann9.1.0-a3-B105` (A3 Series) and `cann9.1.0-950-B105` (950PR/DT Series).
For details, see [Docker image versions](/docs/hardware-platforms/ascend-npus/faq#8-docker-image-versions-stable-release-vs-daily-build).

{/* TODO: Interim explanation (CANN 9.1.0 has no stable release yet). Remove it and switch the tags below to the stable release once it is published. */}

<Tabs>
  <Tab title="A3 Series">

```bash Command
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-B105

docker run -itd --shm-size=16g --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0  \
--device=/dev/davinci1:/dev/davinci1  \
--device=/dev/davinci2:/dev/davinci2  \
--device=/dev/davinci3:/dev/davinci3  \
--device=/dev/davinci4:/dev/davinci4  \
--device=/dev/davinci5:/dev/davinci5  \
--device=/dev/davinci6:/dev/davinci6  \
--device=/dev/davinci7:/dev/davinci7  \
--device=/dev/davinci8:/dev/davinci8  \
--device=/dev/davinci9:/dev/davinci9  \
--device=/dev/davinci10:/dev/davinci10  \
--device=/dev/davinci11:/dev/davinci11  \
--device=/dev/davinci12:/dev/davinci12  \
--device=/dev/davinci13:/dev/davinci13  \
--device=/dev/davinci14:/dev/davinci14  \
--device=/dev/davinci15:/dev/davinci15  \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-B105
```

  </Tab>
  <Tab title="950PR/DT Series">

```bash Command
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-950-B105

docker run -itd --shm-size=16g --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0  \
--device=/dev/davinci1:/dev/davinci1  \
--device=/dev/davinci2:/dev/davinci2  \
--device=/dev/davinci3:/dev/davinci3  \
--device=/dev/davinci4:/dev/davinci4  \
--device=/dev/davinci5:/dev/davinci5  \
--device=/dev/davinci6:/dev/davinci6  \
--device=/dev/davinci7:/dev/davinci7  \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-950-B105
```

  </Tab>
</Tabs>

<Tip>
- If the model weights have already been downloaded to a shared directory, use `-v` to mount the model path into the
  container, for example: `-v /path/to/models:/models`.
- Replace `${NAME}` with your own container name or remove `--name` to use default name.
</Tip>

## Online service deployment

### Single-node PD mixed deployment

Single-node deployment completes both prefill and decode within the same node (PD mixed mode), suitable for scenarios
with limited hardware resources. Select the script for your platform and run it inside the container.

<Tabs>
  <Tab title="A3 Series">

DeepSeek-V4-Flash can be launched with half of an A3 Series node (4 cards, 8 dies). The configuration below was
validated with `--tp-size 8`.

```bash Command
# ============================================================
# Before running, update the following variable:
#   MODEL_PATH: path to the model weights directory
# ============================================================

MODEL_PATH=/path/to/model-weights

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_BUFFSIZE=8
export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:14699"
export SGLANG_ZBAL_LOCAL_MEM_SIZE=62084
export STREAMS_PER_DEVICE=32
export ZBAL_ENABLE_GRAPH=1
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 8 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.8 \
    --prefill-max-requests 160 \
    --max-prefill-tokens 80000 \
    --chunked-prefill-size 131072 \
    --max-running-requests 160 \
    --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --enable-dp-lm-head \
    --kv-cache-dtype bfloat16
```

  </Tab>
  <Tab title="950PR/DT Series">

The 950PR/DT Series (SoC name A5) exposes one device per card, and DeepSeek-V4-Flash can be launched with half of a
node (4 cards). Use the native `DeepSeek-V4-Flash-0731` checkpoint. The configuration below was validated with
`--tp-size 4` and DSPARK speculative decoding.

```bash Command
# ============================================================
# Before running, update the following variable:
#   MODEL_PATH: path to the model weights directory
# ============================================================

MODEL_PATH=/path/to/model-weights

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export DEEPEP_HCCL_BUFFSIZE=1600
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export FORCE_DRAFT_MODEL_NON_QUANT=1
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.72 \
    --max-running-requests 64 \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --enable-dp-lm-head \
    --disable-radix-cache \
    --enable-dp-attention --dp-size 4 \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5
```

  </Tab>
</Tabs>

For the complete benchmark data and additional single-node configurations, see
[DeepSeek-V4-Flash Best Practice — PD Mixed On A3 Series](/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash#single-node-pd-mixed).

### Multi-node PD disaggregation deployment

PD disaggregation splits the prefill and decode stages onto separate nodes, reducing interference and improving
throughput for high-concurrency scenarios. This scenario is already covered in the best practice. For the complete, optimized
deployment commands and benchmark data, see
[DeepSeek-V4-Flash Best Practice — PD Disaggregation On A3 Series](/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash#pd-disaggregation).

## Functional verification

After the service is started, you can invoke the model by sending a prompt:

```shell
# ============================================================
# Before running, update the following variables:
#   HOST: the server host address (e.g., localhost)
#   PORT: the server port number (e.g., 6688)
# ============================================================

curl http://${HOST}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    }'
```

Expected result: an HTTP 200 response with the generated text containing "Paris".

Once the server prints `The server is fired up and ready to roll!` in the logs, it is ready to accept requests. For more
testing examples (Health Check, Generate, Chat Completions, and port usage guidance),
see [Testing the Service](/docs/hardware-platforms/ascend-npus/getting-started/installation#testing-the-service).

## Accuracy evaluation

For accuracy evaluation methods and datasets, see [Accuracy Evaluation on Ascend NPU](/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation).

## Performance

For performance data and benchmark commands, see [Performance Testing on Ascend NPU](/docs/hardware-platforms/ascend-npus/evaluation/performance_testing).

## Best practices

### Best practice configuration reference

For complete optimal configurations with deployment scripts and benchmark commands, see the
[DeepSeek-V4-Flash Best Practice](/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash) page.

## Performance tuning

For the full list of supported features, see [Supported features](#supported-features). For detailed optimization
guidance, see [Optimization on Ascend NPU](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning).

## FAQ

For common environment, installation, and general parameter issues, please refer to the [Ascend NPU FAQ](/docs/hardware-platforms/ascend-npus/faq).
