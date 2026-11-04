---
title: "GLM-5.3"
metatags:
  description: "Deploy GLM-5.3 on Ascend NPUs with SGLang, covering single-node, multi-node, and prefill-decode disaggregation deployments."
---

## Introduction

GLM-5.3 is a large-scale Mixture-of-Experts (MoE) language model in the GLM (General Language Model) series, developed
by Zhipu AI. It uses DeepSeek Sparse Attention (DSA), supports multi-token prediction (MTP), and is designed for
long-context text generation. GLM-5.3 is distinct from the separately released GLM-5.3-Flash model, which uses a
hybrid sparse/linear-attention architecture.

This document demonstrates the deployment of GLM-5.3 on Ascend NPUs using SGLang, including single-node deployment,
multi-node co-located (mixed) deployment, prefill-decode disaggregation, feature configuration, and performance optimization.
The multi-node and prefill-decode disaggregation examples use a locally prepared `GLM-5.3-w8a8` checkpoint and a CANN 9.1.0 runtime.

## Supported features

| Feature                       | Example usage                                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------------------------|
| Tensor Parallelism            | `--tp-size 16` or `--tp-size 32`                                                                          |
| Data Parallelism              | `--dp-size 16` (mixed) or `--dp-size 32` (decode) with `--enable-dp-attention`                                                      |
| Expert Parallelism            | `--ep-size 32` with `--moe-a2a-backend deepep` and `--deepep-mode low_latency`                             |
| Pipeline Parallelism          | `--pp-size 4` with `SGLANG_PP_LAYER_PARTITION=18,20,24,16`                                                |
| Prefill Context Parallelism   | `--attn-cp-size 8` with `--enable-prefill-cp` and `--cp-strategy interleave`                                |
| PD Disaggregation             | `--disaggregation-mode prefill` / `decode` with `--disaggregation-transfer-backend ascend`                  |
| Quantization                  | `--quantization modelslim`                                                                               |
| Chunked Prefill               | `--chunked-prefill-size 65536` / `--max-prefill-tokens 280000` (mixed deployment)                    |
| NPU Graph                     | `--cuda-graph-bs-decode 2` (mixed); `--disable-cuda-graph` (prefill); `--cuda-graph-bs-decode 1 2 3` (decode)                               |
| Speculative Decoding          | `--speculative-algorithm NEXTN` with `--speculative-num-steps 3` and `--speculative-num-draft-tokens 4`     |
| Overlap Schedule              | `export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`                                                             |
| DP LM Head                    | `--enable-dp-lm-head`                                                                                    |

<Note>
The values in the **Example usage** column reflect the deployment scenarios described below and are not universal
defaults. Some flags require a specific SGLang version or model checkpoint. For parameter details, see
[Feature descriptions](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning#feature-descriptions); for
recommended configurations, see [Best practices](#best-practices).
</Note>

For feature compatibility and conflict information between features,
see [Feature Compatibility](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning#feature-compatibility).

## Prerequisites

### Environment

Before following this tutorial, complete the environment setup in the documents below:

- [Ascend NPU Quickstart](/docs/hardware-platforms/ascend-npus/getting-started/quick_start) — the fastest way to get started.
  It walks you through launching a container, starting the SGLang server, and sending a test request.
- [SGLang Installation with NPU Support](/docs/hardware-platforms/ascend-npus/getting-started/installation) — the full installation guide.
  It covers version compatibility (CANN, TorchNPU, Triton, kernels, etc.), building from source or from a Dockerfile,
  and recommended system settings (CPU power scheme, NUMA, swap).

The mixed and PD disaggregation scripts below require a CANN **9.1.0** environment, the corresponding ATB and customized
operator libraries, a compatible SGLang source checkout, and working inter-node HCCL/Gloo connectivity. Ensure that
all participating nodes use compatible software versions and can access the model weights. The PD topology additionally requires the configured MemFabric components.

### Model weights

Before downloading model weights, check the model size to reserve enough disk space.
For multi-node deployment, download the weights to a shared directory accessible to all nodes or distribute an
identical checkpoint to the same local path on every node.

- [GLM-5.3-BF16](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3-BF16) (BF16, approximately 1.51 TB).
- `GLM-5.3-w8a8` (a locally prepared W8A8 ModelSlim-quantized checkpoint used in the scripts below; set `MODEL_PATH`
  to the actual location). The checkpoint size and MTP-weight availability depend on the conversion output.
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize model weights for Ascend devices.

<Info>
The deployment examples below use a W8A8 checkpoint with `--quantization modelslim`. The PD disaggregation topology
uses **two Atlas 800I A3 nodes for prefill and two additional A3 nodes for decode**. Make sure that the checkpoint
contains the required MTP weights before enabling NEXTN speculative decoding.
</Info>

For hardware specifications (memory per device, devices per node, and differences between A2 and A3),
see [Ascend NPU Reference — Hardware](/docs/hardware-platforms/ascend-npus/reference/glossary#hardware).

## Installation

<Warning>
Ensure sufficient disk space before pulling images. A CANN Docker image requires at least **30GB** of free space.
CANN 9.1.0 base images do **not** by themselves include a ready-to-run SGLang environment: install a compatible
SGLang build, TorchNPU, ATB, custom operators, and other required dependencies before running the deployment scripts.
</Warning>

The commands below start CANN 9.1.0 base images. Follow
[SGLang Installation with NPU Support](/docs/hardware-platforms/ascend-npus/getting-started/installation)
to install SGLang and the required runtime components inside the container. Use the matching image for your hardware.

<Tabs>
  <Tab title="Atlas 800I A3">

```bash Command
docker pull quay.io/ascend/sglang:cann9.1.0-a3-B080

docker run -itd --shm-size=16g --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci1:/dev/davinci1 \
--device=/dev/davinci2:/dev/davinci2 \
--device=/dev/davinci3:/dev/davinci3 \
--device=/dev/davinci4:/dev/davinci4 \
--device=/dev/davinci5:/dev/davinci5 \
--device=/dev/davinci6:/dev/davinci6 \
--device=/dev/davinci7:/dev/davinci7 \
--device=/dev/davinci8:/dev/davinci8 \
--device=/dev/davinci9:/dev/davinci9 \
--device=/dev/davinci10:/dev/davinci10 \
--device=/dev/davinci11:/dev/davinci11 \
--device=/dev/davinci12:/dev/davinci12 \
--device=/dev/davinci13:/dev/davinci13 \
--device=/dev/davinci14:/dev/davinci14 \
--device=/dev/davinci15:/dev/davinci15 \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
quay.io/ascend/sglang:cann9.1.0-a3-B080
```

  </Tab>
  <Tab title="Atlas 800I A2">

```bash Command
docker pull quay.io/ascend/sglang:cann9.1.0-a3-B080

docker run -itd --shm-size=16g --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci1:/dev/davinci1 \
--device=/dev/davinci2:/dev/davinci2 \
--device=/dev/davinci3:/dev/davinci3 \
--device=/dev/davinci4:/dev/davinci4 \
--device=/dev/davinci5:/dev/davinci5 \
--device=/dev/davinci6:/dev/davinci6 \
--device=/dev/davinci7:/dev/davinci7 \
--device=/dev/davinci8:/dev/davinci8 \
--device=/dev/davinci9:/dev/davinci9 \
--device=/dev/davinci10:/dev/davinci10 \
--device=/dev/davinci11:/dev/davinci11 \
--device=/dev/davinci12:/dev/davinci12 \
--device=/dev/davinci13:/dev/davinci13 \
--device=/dev/davinci14:/dev/davinci14 \
--device=/dev/davinci15:/dev/davinci15 \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
quay.io/ascend/sglang:cann9.1.0-a3-B080
```

  </Tab>
</Tabs>

<Tip>
- If the model weights have already been downloaded to a shared directory, use `-v` to mount the model path into the
  container, for example: `-v /path/to/models:/models`.
- Replace `${NAME}` with your container name or remove `--name` to use the default name.
- The listed `/dev/davinci*` mappings follow the 16-device A3 example in the original tutorial. Remove mappings for
  devices that do not exist on your host, particularly when using A2 nodes.
- If you already have a validated CANN 9.1.0 SGLang development image, you can use it instead of building from the base image.
</Tip>

## Online service deployment

### Single-node deployment

The following single-node example is adapted from the GLM-5.2 tutorial and is provided as a
**reference configuration**.

```shell
#!/usr/bin/env bash
# Reference only: adapted from the GLM-5.2 tutorial; not yet verified for GLM-5.3.
# Update MODEL_PATH, SGLANG_ROOT, and the installed custom-op paths as required.
MODEL_PATH='/path/to/GLM-5.3-w8a8'
SGLANG_ROOT='/path/to/sglang'

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1000
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_NPU_USE_MULTI_STREAM=1

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend ascend --device npu \
    --tp-size 16 --nnodes 1 --node-rank 0 \
    --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
    --trust-remote-code \
    --host 127.0.0.1 --port 8000 \
    --mem-fraction-static 0.75 --max-running-requests 32 \
    --served-model-name glm-5 \
    --quantization modelslim --dtype bfloat16 \
    --moe-a2a-backend deepep --deepep-mode auto \
    --disable-shared-experts-fusion \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

### Multi-node deployment

For co-located (mixed) serving across **two Atlas 800I A3 nodes**, both nodes participate in a single GLM-5.3
inference instance. The following script is adapted from the supplied GLM-5.3 W8A8 multi-node configuration, using
**TP=32**, **DP=16**, DeepEP `auto`, and NEXTN speculative decoding.

Modify the node IP addresses, model/source paths, and network interface, then run the **same script on both nodes**.
The two nodes must use the same compatible runtime and be able to communicate with rank 0 via the distributed
initialization address. In this script, the mixed-serving HTTP port is **6688** (not 8000).

```shell
#!/usr/bin/env bash
# GLM-5.3 W8A8: two-node co-located (mixed) deployment (Atlas 800I A3).
# Run the same script on both nodes after replacing the variables below.

# ============================================================
# Before running, update the following variables:
#   IPS: IP addresses of two nodes (rank 0 first)
#   NIC: network interface name for HCCL and Gloo
#   MODEL_PATH: GLM-5.3 W8A8 weight directory
#   SGLANG_ROOT: compatible SGLang source checkout
# ============================================================
IPS=('<node-0-ip>' '<node-1-ip>')
NIC='<network-interface>'
MODEL_PATH='/path/to/GLM-5.3-w8a8'
SGLANG_ROOT='/path/to/sglang'

# High-performance CPU settings (requires sufficient permissions).
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"

# Keep the allocator configuration disabled as in the supplied script.
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME="${NIC}"
export GLOO_SOCKET_IFNAME="${NIC}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=40
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# Optional tuning flags from the original script (disabled by default).
# export SGLANG_NPU_USE_MLAPO=1
# export HCCL_OP_EXPANSION_MODE=AIV
# export SGLANG_NPU_USE_MULTI_STREAM=1
# export SGLANG_USE_AG_AFTER_QLORA=1
# export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

# Match the configured node IP against all local IPv4 addresses.
LOCAL_IPS=( $(hostname -I) )
has_local_ip() {
    local wanted="$1" local_ip
    for local_ip in "${LOCAL_IPS[@]}"; do
        [[ "${wanted}" == "${local_ip}" ]] && return 0
    done
    return 1
}

for i in "${!IPS[@]}"; do
    if has_local_ip "${IPS[$i]}"; then
        echo "Starting GLM-5.3 mixed deployment on ${IPS[$i]} (rank ${i})"
        export HCCL_BUFFSIZE=2000
        exec python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --attention-backend ascend \
            --device npu \
            --dist-init-addr "${IPS[0]}:5000" \
            --tp-size 32 --nnodes 2 --node-rank "$i" \
            --dp-size 16 --enable-dp-attention \
            --chunked-prefill-size 65536 --max-prefill-tokens 280000 \
            --trust-remote-code \
            --host "${IPS[$i]}" \
            --mem-fraction-static 0.68 \
            --port 6688 \
            --served-model-name glm-5 \
            --cuda-graph-bs-decode 2 \
            --max-running-requests 32 \
            --quantization modelslim \
            --moe-a2a-backend deepep --deepep-mode auto \
            --load-balance-method round_robin \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4
    fi
done

echo "ERROR: Neither IPS address matches this host: ${LOCAL_IPS[*]}" >&2
exit 1
```

### Prefill-decode disaggregation deployment

PD disaggregation splits the prefill and decode stages onto separate node groups, reducing interference between
long-prompt processing and autoregressive generation. This configuration uses **four Atlas 800I A3 nodes**:

- **Prefill group:** 2 nodes, `TP=8`, `PP=4`, `DP=1`, and prefill context parallelism `CP=8`. It runs DeepEP in `normal`
  mode and disables NPU graph capture.
- **Decode group:** 2 nodes, `TP=32`, `DP=32`, `EP=32`. It runs DeepEP in `low_latency` mode with decode graph capture
  and NEXTN speculative decoding.

Run the following **same script on all four nodes**. The script selects the role and node rank by matching a configured
IP address against the local network addresses. Update the model/source paths, the two prefill and two decode IP
addresses, and the HCCL/Gloo interface before running it. The nodes must reach the prefill master for MemFabric
bootstrap and should be started before the router.

```shell
#!/usr/bin/env bash
# GLM-5.3 W8A8: 2 prefill nodes + 2 decode nodes (Atlas 800I A3).
# Run this same script on all four nodes, after replacing the placeholders below.
# The IP addresses must match addresses returned by `hostname -I` on each node.

# ============================================================
# Before running, update the following variables:
#   P_IP: two prefill node IP addresses (rank 0 first)
#   D_IP: two decode node IP addresses (rank 0 first)
#   NIC: HCCL/Gloo network interface name
#   MODEL_PATH: local/shared GLM-5.3 W8A8 model directory
#   SGLANG_ROOT: SGLang source directory
# Ensure MemFabric, ATB, and CANN 9.1.0 custom ops are installed.
# ============================================================
P_IP=('<prefill-node-0-ip>' '<prefill-node-1-ip>')
D_IP=('<decode-node-0-ip>' '<decode-node-1-ip>')
NIC='<network-interface>'
MODEL_PATH='/path/to/GLM-5.3-w8a8'
SGLANG_ROOT='/path/to/sglang'

# High-performance CPU configuration (requires sufficient permissions).
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_MF_LOG_LEVEL=2
export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24707"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export HCCL_SOCKET_IFNAME="${NIC}"
export GLOO_SOCKET_IFNAME="${NIC}"

# Test every local interface rather than assuming the relevant IP is first.
LOCAL_IPS=( $(hostname -I) )
has_local_ip() {
    local wanted="$1" local_ip
    for local_ip in "${LOCAL_IPS[@]}"; do
        [[ "${wanted}" == "${local_ip}" ]] && return 0
    done
    return 1
}

# Prefill: two A3 nodes (32 NPU devices in total); TP=8, PP=4, CP=8.
for i in "${!P_IP[@]}"; do
    if has_local_ip "${P_IP[$i]}"; then
        export HCCL_BUFFSIZE=1200
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export ENABLE_PROFILING=0
        export SGLANG_PP_LAYER_PARTITION=18,20,24,16

        exec python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode prefill \
            --host "${P_IP[$i]}" --port 8000 \
            --disaggregation-bootstrap-port 8998 \
            --dist-init-addr "${P_IP[0]}:5000" \
            --trust-remote-code --nnodes 2 --node-rank "$i" \
            --mem-fraction-static 0.72 \
            --attention-backend ascend --device npu \
            --quantization modelslim --dtype bfloat16 \
            --tp-size 8 --pp-size 4 --dp-size 1 \
            --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave \
            --enable-dp-attention --enable-dp-lm-head \
            --moe-dense-tp 1 --load-balance-method round_robin \
            --disaggregation-transfer-backend ascend \
            --max-running-requests 64 \
            --served-model-name glm-5 \
            --chunked-prefill-size 4096 --max-prefill-tokens 180000 \
            --moe-a2a-backend deepep --deepep-mode normal \
            --disable-shared-experts-fusion --disable-cuda-graph \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 1 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 2
    fi
done

# Decode: two A3 nodes (32 NPU devices in total); TP=32, DP=32, EP=32.
for i in "${!D_IP[@]}"; do
    if has_local_ip "${D_IP[$i]}"; then
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export DEEPEP_HCCL_BUFFSIZE=200
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
        export TASK_QUEUE_ENABLE=0
        export SGLANG_NPU_USE_MULTI_STREAM=1

        exec python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode decode \
            --host "${D_IP[$i]}" --port 8003 \
            --trust-remote-code \
            --dist-init-addr "${D_IP[0]}:5000" \
            --nnodes 2 --node-rank "$i" \
            --tp-size 32 --dp-size 32 --enable-dp-attention --ep-size 32 \
            --mem-fraction-static 0.97 --max-running-requests 32 \
            --attention-backend ascend --device npu \
            --quantization modelslim --dtype bfloat16 \
            --served-model-name glm-5 \
            --moe-a2a-backend deepep --deepep-mode low_latency \
            --cuda-graph-bs-decode 1 2 3 \
            --disaggregation-transfer-backend ascend \
            --watchdog-timeout 9000 --context-length 180000 \
            --tokenizer-worker-num 4 \
            --disable-shared-experts-fusion \
            --load-balance-method round_robin \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4
    fi
done

echo "ERROR: None of the P_IP/D_IP addresses matches this host: ${LOCAL_IPS[*]}" >&2
exit 1
```

Launch the router after the prefill and decode services are ready. The router example below preserves the
GLM-5.2 document's PD routing syntax and uses the prefill/decode group master addresses.

```shell
# ============================================================
# Before running, update the following variables:
#   P_MASTER_IP: first prefill node IP (prefill rank 0)
#   D_MASTER_IP: first decode node IP (decode rank 0)
#   ROUTER_HOST_IP: router listening IP
# ============================================================
P_MASTER_IP="<prefill-node-0-ip>"
D_MASTER_IP="<decode-node-0-ip>"
ROUTER_HOST_IP="<router-host-ip>"

python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy round_robin \
    --prefill http://${P_MASTER_IP}:8000 8998 \
    --decode http://${D_MASTER_IP}:8003 \
    --host "${ROUTER_HOST_IP}" \
    --port 6688
```

<Note>
The prefill and decode groups use separate `--dist-init-addr` endpoints. The `ASCEND_MF_STORE_URL` must point to the
prefill rank-0 node. The `--max-prefill-tokens 180000` and `--context-length 180000` settings describe this example
configuration, **not** the model's maximum supported context. For troubleshooting, turn off optional overlap or
multi-stream settings one at a time to isolate compatibility issues.
</Note>

## Functional verification

After the service is started, invoke the model by sending a prompt. For PD disaggregation, use the router address and
port `6688`; for single-node co-located serving, use port `8000`, and for the two-node mixed deployment, use port `6688`.

```shell
# ============================================================
# Before running, update the following variables:
#   HOST: router/server host IP address
#   PORT: 6688 for the PD router or mixed deployment; 8000 for single-node mode
# ============================================================
HOST="<router-or-server-ip>"
PORT=6688

curl http://${HOST}:${PORT}/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "What is the capital of France?",
        "sampling_params": {
            "max_new_tokens": 64,
            "temperature": 0
        }
    }'
```

Expected result: an HTTP 200 response containing the generated answer (for example, "Paris").

Once the server prints `The server is fired up and ready to roll!` in the logs, it is ready to accept requests. For
additional testing examples (Health Check, Generate, Chat Completions, and port usage),
see [Testing the Service](/docs/hardware-platforms/ascend-npus/getting-started/installation#testing-the-service).

## Accuracy evaluation

For accuracy evaluation methods and datasets, see [Accuracy Evaluation on Ascend NPU](/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation).

## Performance

For performance data and benchmark commands, see [Performance Testing on Ascend NPU](/docs/hardware-platforms/ascend-npus/evaluation/performance_testing).

## Best practices

### Best practice configuration reference

GLM-5.3 does not have a validated standalone best-practice page in this tutorial. The
[Online service deployment](#online-service-deployment) section above includes a four-node PD disaggregation
configuration and a two-node mixed deployment configuration based on supplied GLM-5.3 scripts. The single-node
co-located example remains a GLM-5.2-derived reference. The configurations have not been independently verified on hardware.

The two-node mixed example uses `TP=32`, `DP=16`, `HCCL_BUFFSIZE=2000`, and `--cuda-graph-bs-decode 2`.
Disable profiling when measuring uninstrumented service performance.

For this PD disaggregation configuration, prefill prioritizes long sequences using pipeline/context parallelism and
DeepEP `normal`, while decode prioritizes low latency using DP/EP, DeepEP `low_latency`, NPU multi-stream,
and NEXTN. Tune `--mem-fraction-static`, graph capture batch sizes, and concurrency based on observed HBM usage,
TTFT, TPOT, and request throughput. Check that the weight conversion and MTP settings are compatible.

For the full catalog of optimization features and their parameter and compatibility details,
see [Optimization on Ascend NPU](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning).

## Performance tuning

For supported features, see [Supported features](#supported-features). For detailed optimization
guidance, see [Optimization on Ascend NPU](/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning).

## FAQ

For common environment, installation, and parameter issues, refer to the [Ascend NPU FAQ](/docs/hardware-platforms/ascend-npus/faq).
