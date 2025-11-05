# Ascend NPUs

You can install SGLang using any of the methods below. Please go through `System Settings` section to ensure the clusters are roaring at max performance. Feel free to leave an issue [here at sglang](https://github.com/sgl-project/sglang/issues) if you encounter any issues or have any problems.

## System Settings

### CPU performance power scheme

The default power scheme on Ascend hardware is `ondemand` which could affect performance, changing it to `performance` is recommended.

```shell
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make sure changes are applied successfully
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor # shows performance
```

### Disable NUMA balancing

```shell
sudo sysctl -w kernel.numa_balancing=0

# Check
cat /proc/sys/kernel/numa_balancing # shows 0
```

### Prevent swapping out system memory

```shell
sudo sysctl -w vm.swappiness=10

# Check
cat /proc/sys/vm/swappiness # shows 10
```

## Installing SGLang

### Method 1: Installing from source with prerequisites

#### Python Version

Only `python==3.11` is supported currently. If you don't want to break system pre-installed python, try installing with [conda](https://github.com/conda/conda).

```shell
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

#### MemFabric Adaptor

_TODO: MemFabric is still a working project yet open sourced til August/September, 2025. We will release it as prebuilt wheel package for now._

_Notice: Prebuilt wheel package is based on `aarch64`, please leave an issue [here at sglang](https://github.com/sgl-project/sglang/issues) to let us know the requests for `amd64` build._

MemFabric Adaptor is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```shell
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${MF_WHL_NAME}"
wget -O "${MF_WHL_NAME}" "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"
```

#### Pytorch and Pytorch Framework Adaptor on Ascend

Only `torch==2.6.0` is supported currently due to NPUgraph and Triton-on-Ascend's limitation, however a more generalized version will be release by the end of September, 2025.

```shell
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu

PTA_VERSION="v7.1.0.1-pytorch2.6.0"
PTA_NAME="torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl"
PTA_URL="https://gitee.com/ascend/pytorch/releases/download/${PTA_VERSION}/${PTA_WHL_NAME}"
wget -O "${PTA_NAME}" "${PTA_URL}" && pip install "./${PTA_NAME}"
```

#### vLLM

vLLM is still a major prerequisite on Ascend NPU. Because of `torch==2.6.0` limitation, only vLLM v0.8.5 is supported.

```shell
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG
(cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v -e .)
```

#### Triton on Ascend

_Notice:_ We recommend installing triton-ascend from source due to its rapid development, the version on PYPI can't keep up for know. This problem will be solved on Sep. 2025, afterwards `pip install` would be the one and only installing method.

Please follow Triton-on-Ascend's [installation guide from source](https://gitee.com/ascend/triton-ascend#2%E6%BA%90%E4%BB%A3%E7%A0%81%E5%AE%89%E8%A3%85-triton-ascend) to install the latest `triton-ascend` package.

#### DeepEP-compatible Library

We are also providing a DeepEP-compatible Library as a drop-in replacement of deepseek-ai's DeepEP library, check the [installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

#### Installing SGLang from source

```shell
# Use the last release branch
git clone -b v0.5.4.post3 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e python[srt_npu]
```

### Method 2: Using docker

__Notice:__ `--privileged` and `--network=host` are required by RDMA, which is typically needed by Ascend NPU clusters.

__Notice:__ The following docker command is based on Atlas 800I A3 machines. If you are using Atlas 800I A2, make sure only `davinci[0-7]` are mapped into container.

```shell
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t <image_name> -f npu.Dockerfile .

alias drun='docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/'

drun --env "HF_TOKEN=<secret>" \
    <image_name> \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend --host 0.0.0.0 --port 30000
```

## Examples

### Running DeepSeek-V3

Running DeepSeek with PD disaggregation on 2 x Atlas 800I A3.
Model weights could be found [here](https://modelers.cn/models/State_Cloud/Deepseek-R1-bf16-hfd-w8a8).

Prefill:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"

drun <image_name> \
    python3 -m sglang.launch_server --model-path State_Cloud/DeepSeek-R1-bf16-hfd-w8a8 \
    --trust-remote-code \
    --attention-backend ascend \
    --mem-fraction-static 0.8 \
    --quantization w8a8_int8 \
    --tp-size 16 \
    --dp-size 1 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 6657 \
    --disaggregation-transfer-backend ascend \
    --dist-init-addr <PREFILL_HOST_IP>:6688 \
    --host <PREFILL_HOST_IP> \
    --port 8000
```

Decode:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"
export HCCL_BUFFSIZE=200
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=24
export SGLANG_NPU_USE_MLAPO=1

drun <image_name> \
    python3 -m sglang.launch_server --model-path State_Cloud/DeepSeek-R1-bf16-hfd-w8a8 \
    --trust-remote-code \
    --attention-backend ascend \
    --mem-fraction-static 0.8 \
    --quantization w8a8_int8 \
    --enable-deepep-moe \
    --deepep-mode low_latency \
    --tp-size 16 \
    --dp-size 1 \
    --ep-size 16 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --dist-init-addr <DECODE_HOST_IP>:6688 \
    --host <DECODE_HOST_IP> \
    --port 8001
```

Mini_LB:

```shell
drun <image_name> \
    python -m sglang.srt.disaggregation.launch_lb \
    --prefill http://<PREFILL_HOST_IP>:8000 \
    --decode http://<DECODE_HOST_IP>:8001 \
    --host 127.0.0.1 --port 5000
```
