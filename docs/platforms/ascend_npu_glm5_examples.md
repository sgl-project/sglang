# GLM-5 examples

## Introduction

The GLM (General Language Model) series is an open-source bilingual large language model family jointly developed by the KEG Laboratory of Tsinghua University and Zhipu AI. This series of models has performed outstandingly in the field of Chinese NLP with its unique unified pre-training framework and bilingual capabilities. [GLM-5](https://huggingface.co/zai-org/GLM-5) adopts the DeepSeek-V3/V3.2 architecture, including the sparse attention (DSA) and multi-token prediction (MTP). Ascend supports GLM-5 with 0Day based on the SGLang inference framework, achieving low-code seamless enablement and compatibility with the mainstream distributed parallel capabilities within the current SGLang framework. We welcome developers to download and experience it.

## Environment Preparation

### Model Weight

- `GLM-5.0`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5).
- `GLM-5.0-w4a8`(Quantized version without mtp): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5-w4a8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantify the model naively.


### Installation

The dependencies required for the NPU runtime environment have been integrated into a Docker image and uploaded to the quay.io platform. You can directly pull it.

```{code-block} bash
#Atlas 800 A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-glm5
#Atlas 800 A2
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-910b-glm5

#start container
docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0  \
--device=/dev/davinci1:/dev/avinci1  \
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
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:${TAG}
```

Note: Using this image, you need to update transformers to main branch
``` shell
# reinstall transformers
pip install git+https://github.com/huggingface/transformers.git
```

## Deployment

### Single-node Deployment

- Quantized model `glm5_w4a8` can be deployed on 1 Atlas 800 A3 (64G × 16) .

Run the following script to execute online inference.

```shell
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size 16384 --max-prefill-tokens 280000 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.7 \
        --port 8000 \
        --served-model-name glm-5 \
        --cuda-graph-bs 16 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto
```

### Multi-node Deployment

- `GLM-5-bf16`: require at least 2 Atlas 800 A3 (64G × 16).

**A3 series**

Modify the IP of 2 nodes, then run the same scripts on two nodes.

**node 0/1**

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV

# Run command ifconfig on two nodes, find out which inet addr has same IP with your node IP. That is your public interface, which should be added here
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo


P_IP=('your ip1' 'your ip2')
P_MASTER="${P_IP[0]}:your port"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 32 --nnodes 2 --node-rank $i --dist-init-addr $P_MASTER \
        --chunked-prefill-size 16384 --max-prefill-tokens 131072 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.8\
        --port 8000 \
        --served-model-name glm-5 \
        --cuda-graph-max-bs 16 \
        --disable-radix-cache
        NODE_RANK=$i
        break
    fi
done

```

### Prefill-Decode Disaggregation

Not test yet.

### Using Benchmark

Refer to [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md) for details.
