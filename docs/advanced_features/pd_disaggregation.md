# PD Disaggregation

## Why and What is PD Disaggregation?

Large Language Model (LLM) inference comprises two distinct phases: **Prefill** and **Decode**. The Prefill phase is computation-intensive, processing the entire input sequence, while the Decode phase is memory-intensive, managing the Key-Value (KV) cache for token generation. Traditionally, these phases are handled within a unified engine, where combined scheduling of prefill and decode batches introduces inefficiencies. To address these challenges, we introduce **Prefill and Decoding (PD) Disaggregation** in SGLang.

### Issues with Unified Scheduling

The conventional unified engine, which processes prefill and decode batches together, results in two significant problems:

1. **Prefill Interruption**: Incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation.
2. **DP Attention Imbalance**: In data-parallel (DP) attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency.

PD Disaggregation resolves these by separating the two stages, enabling tailored optimizations for each.

For the design details, please refer to [link](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0).

Currently, we support Mooncake and NIXL as the transfer engine.

## Profiling in PD Disaggregation Mode

When you need to profile prefill or decode workers in PD disaggregation mode, please refer to the [Profile In PD Disaggregation Mode](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html#profile-in-pd-disaggregation-mode) section in the Benchmark and Profiling guide. Due to torch profiler limitations, prefill and decode workers must be profiled separately using dedicated command-line options.

## Router Integration

For deploying PD disaggregation at scale with load balancing and fault tolerance, SGLang provides a router. The router can distribute requests between prefill and decode instances using various routing policies. For detailed information on setting up routing with PD disaggregation, including configuration options and deployment patterns, see the [SGLang Router documentation](router.md#mode-3-prefill-decode-disaggregation).


## Mooncake
### Requirements

```bash
uv pip install mooncake-transfer-engine
```

### Usage

### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```
### Advanced Configuration

PD Disaggregation with Mooncake supports the following environment variables for fine-grained control over system behavior.

#### NVLink Transport Configuration
To enable NVLink transport for KV cache transfers with the mooncake backend (recommended for NVL72 deployments), set the following environment variables. Note that auxiliary data transfer will still use TCP as a temporary workaround.

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
export MC_FORCE_MNNVL=True
```

#### Prefill Server Configuration
| Variable | Description | Default |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`** | Controls the total number of worker threads for KVCache transfer operations per TP rank | A dynamic value calculated by `int(0.75 * os.cpu_count()) // 8)`, which is limited to be larger than 4 and less than 12 to ensure efficiency and prevent thread race conditions |
| **`SGLANG_DISAGGREGATION_QUEUE_SIZE`** | Sets the number of parallel transfer queues. KVCache transfer requests from multiple decode instances will be sharded into these queues so that they can share the threads and the transfer bandwidth at the same time. If it is set to `1`, then we transfer requests one by one according to fcfs strategy | `4` |
| **`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`** | Timeout (seconds) for receiving destination KV indices during request initialization | `300` |

If a greater mean TTFT is acceptable, you can `export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600` (10 minutes) to relax the timeout condition.
Please be aware that this setting will cause prefill instances to take a longer time to clean up the affected memory resources when a running decode node loses connection.

#### Decode Server Configuration
| Variable | Description | Default |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`** | Interval (seconds) between health checks to prefill bootstrap servers | `5.0` |
| **`SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE`** | Consecutive heartbeat failures before marking prefill server offline | `2` |
| **`SGLANG_DISAGGREGATION_WAITING_TIMEOUT`** | Timeout (seconds) for receiving KV Cache after request initialization | `300` |

If a greater mean TTFT is acceptable, you can `export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600` (10 minutes) to relax the timeout condition.


## NIXL
### Requirements

Install via pip.

```bash
pip install nixl
```

Or build from source - may be required if you already have UCX installed.

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
pip install . --config-settings=setup-args="-Ducx_path=/path/to/ucx"
```


### Usage

### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend nixl
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend nixl
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

## ASCEND

### Usage

Use ascend backend with [mf_adapter(download link)](https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl?AccessKeyId=HPUAXT4YM0U8JNTERLST&Expires=1783151861&Signature=3j10QDUjqk70enaq8lostYV2bEA%3D) and ASCEND_MF_STORE_URL being set

```bash
pip install mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl --force-reinstall
export ASCEND_MF_STORE_URL="tcp://xxx.xx.xxx.xxx:xxxx"
```
Use mooncake backend, more details can be found in mooncake section.
```bash
export ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true
```
ASCEND_NPU_PHY_ID need to be set in container env
```bash
export ASCEND_NPU_PHY_ID=xxx
```


### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend ascend
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend ascend
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
```
