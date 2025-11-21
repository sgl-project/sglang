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
