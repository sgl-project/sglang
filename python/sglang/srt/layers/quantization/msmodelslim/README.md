Quantization [msModelSlim](https://gitcode.com/Ascend/msit/tree/master/msmodelslim) module.

`--quantization modelslim` flag introduced. To load already quantized models, simply load the model weights and config. Again, if the model has been quantized offline, there's no need to add `--quantization modelslim` argument when starting the engine. The quantization method will be parsed from the downloaded `quant_model_description.json` config.

MsModelSlim was developed in the format of compressed_tensors and includes support for various quantization schemes, such as:
- [x] W4A4 dynamic linear
- [x] W8A8 static linear
- [x] W8A8 dynamic linear 
- [x] W4A8 dynamic MOE
- [x] W8A8 dynamic MOE

Also MsModelSlim module include:
- [x] Automated config detection for modelslim format (without the need to specify --quantization modelslim flag)
- [x] Unit-tests for w4a4 modelslim, w8a8 modelslim

Examples of launch:

server: 
`SGLANG_SET_CPU_AFFINITY=1
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
STREAMS_PER_DEVICE=32
HCCL_BUFFSIZE=1536
ENABLE_ASCEND_MOE_NZ=1
ASCEND_RT_VISIBLE_DEVICES=0,1 
python3 -m sglang.launch_server --device npu --attention-backend ascend --trust-remote-code --tp-size 2 --model-path *model* --port 30088 --mem-fraction-static 0.8 --cuda-graph-max-bs 16`

client: 
`python ./benchmark/gsm8k/bench_sglang.py --num-questions 1319 --port 30088 --data-path ../gsm8k/test.jsonl --parallel 16`

<!-- If this pull request affects model outputs (e.g., changes to the kernel or model forward code), provide accuracy test results. -->
Qwen3-32B-W4A4 from msmodelslim (dynamic)  - Ascend 910B2
<img width="844" height="79" alt="image" src="https://github.com/user-attachments/assets/58ca29dd-f885-4877-9657-88e6a7541017" />

Qwen3-32B-W8A8 from msmodelslim (static) - Ascend 910B4
<img width="835" height="78" alt="image" src="https://github.com/user-attachments/assets/9e0ca923-f76e-45e2-bea1-9699af6a0c43" />

Qwen3-32B-W8A8 from msmodelslim (dynamic) - Ascend 910B2
<img width="836" height="76" alt="image" src="https://github.com/user-attachments/assets/25e5b740-1e4a-449a-9d6b-51cf72e60140" />

Qwen3-30B-W8A8 from msmodelslim (attn - static / mlp - dynamic) - Ascend 910B2
<img width="847" height="78" alt="image" src="https://github.com/user-attachments/assets/74fc4fde-9e67-4028-8e44-6dcc6faf9ebc" />

server:
`sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=1536
export ENABLE_ASCEND_MOE_NZ=1
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
`python3 -m sglang.launch_server --model-path *model* --tp 4 --trust-remote-code --attention-backend ascend --device npu  --host 127.0.0.1 --port 30088 --mem-fraction-static 0.8 --quantization modelslim --moe-a2a-backend deepep --deepep-mode auto`

client: 
`python ./benchmark/gsm8k/bench_sglang.py --num-questions 1319 --port 30088 --data-path ../gsm8k/test.jsonl --parallel 16`

Qwen3-30B-W8A8 from msmodelslim (attn - static / mlp - dynamic) with EP - Ascend 910C 
<img width="947" height="80" alt="image" src="https://github.com/user-attachments/assets/7808366f-33d7-4cca-a9fc-e0ce4167f682" />
