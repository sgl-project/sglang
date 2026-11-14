# Qwen3.8-27B deployment and tuning report on Ascend A2

## 1. Experiment information

| Item | Recorded value |
| --- | --- |
| Author | rangeegnar |
| MeetUp task number / developer ID | Pending assignment; the GitHub username above is not an invented task ID |
| Experiment date | 2026-11-14 (Asia/Shanghai; raw tool logs use UTC) |
| Declared image | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112` |
| Installed SGLang | `0.0.0.dev22121+g0fa9ff55f.d20261112` |
| Tested source commit | `0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9` |
| Hardware | 4 x Ascend 910B3 (A2), 64 GiB HBM per device, TP=4 |
| Device mapping | Physical devices 4-7; container logical devices 0-3 |
| Python / torch / torch_npu | 3.12.13 / 2.10.0+cpu / 2.10.0.post6 |
| Transformers / EvalScope | 5.19.0 / 1.14.1 |
| Working directory | `/workspace/meetup_qwen_20261114` |

The image tag comes from the task specification. Software versions and the source commit were checked inside the provided container; its image tag was not independently inspected from the host. These are measurements on the recorded container commit, not a claim that current upstream `main` has been benchmarked.

### Test protocol

GSM8K uses 200 examples, seed 42, 4-shot prompting, temperature 0, `max_tokens=8192`, evaluation concurrency 16, and `enable_thinking=false`. Accuracy was measured again after enabling MTP. All four evaluations completed 200 requests without errors. The tokenized inputs for all 200 examples were checked to be identical between the two model paths with thinking disabled.

The performance workload uses decimal K: 128,000 input tokens = 115,200 shared-prefix tokens + 12,800 suffix tokens; each response has 1,000 output tokens. At concurrency C, the run contains 4C requests. Before each run, flush the cache and warm only the shared prefix with one generated token. Exclude this warmup from measurement. Run the benchmark with `--warmup-requests 0` and without `--flush-cache`, because this tested version flushes after its built-in warmup.

Validate actual request lengths, successful completions, zero errors, and 90% measured cache hits. A cold-cache diagnostic had only 64.2% cache hits and is excluded from the valid comparison. The 14 valid runs have 128,000 input tokens, 1,000 output tokens, and 115,200 cached tokens per request.

The primary selection below requires **every measured request to have TPOT <= 50 ms**. A separate comparison reports the highest observed throughput under **mean TPOT <= 50 ms**; that weaker criterion does not imply that all requests satisfy the limit. TPOT excludes the first token and is distinct from individual inter-token latency. The threshold is an upper bound, not a requirement to reach exactly 50 ms.

Each configuration was measured once with 4C requests at concurrency C. The search is limited to the recorded concurrency grid; it does not establish a global optimum or uncertainty bounds.

## 2. Deployment and measurements

Run one model server at a time. Save the launch commands below as scripts. In a separate terminal, wait for `http://127.0.0.1:6688/health` to become ready before evaluation. The model paths and Ascend environment paths are those used in this experiment.

### Qwen3.8-27B

The selected configuration enables MTP at concurrency 4: 16/16 requests satisfy TPOT <= 50 ms, mean TPOT 21.76 ms, P99 TPOT 33.97 ms, output throughput 141.60 token/s, and 90.00% cache hits.

#### Complete tuned launch command


```bash
#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset ASCEND_LAUNCH_BLOCKING
export NO_PROXY=localhost,127.0.0.1
export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export TASK_QUEUE_ENABLE=1
export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
exec python3 -m sglang.launch_server \
 --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
 --host 127.0.0.1 --port 6688 --tp-size 4 --nnodes 1 \
 --attention-backend ascend --device npu \
 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
 --mamba-radix-cache-strategy extra_buffer \
 --max-running-requests 16 --max-mamba-cache-size 128 \
 --mem-fraction-static 0.80 --context-length 131072 \
 --cuda-graph-bs-decode 1 2 4 8 12 16 \
 --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

#### Accuracy command

Run after the model server is ready:


```bash
export NO_PROXY=localhost,127.0.0.1
evalscope eval \
 --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
 --api-url http://127.0.0.1:6688/v1 --api-key EMPTY \
 --eval-type openai_api --datasets gsm8k --limit 200 --eval-batch-size 16 --seed 42 \
 --generation-config '{"temperature":0,"max_tokens":8192,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
 --work-dir /workspace/meetup_qwen_20261114/results/gsm8k_bf16_mtp
```


Result: **196/200 (98.00%)**, 200 succeeded, 0 errors. The non-MTP baseline scored 195/200 (97.50%). Original final EvalScope output follows; full stdout and per-example artifacts are in the attached evidence archive.


```text
Overall report table: 
┌─────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model       │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├─────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98%     │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘ 
[0m
2026-11-14 02:15:01 - evalscope - [32mINFO[0m: Overall perf table: 
Model        Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
-----------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
Qwen3.8-27B  GSM8K        200  15.421 s   —           —           23.41 tok/s       689        361 
[0m
2026-11-14 02:15:02 - evalscope - [32mINFO[0m: HTML report generated: /workspace/meetup_qwen_20261114/results/gsm8k_bf16_mtp/20261114_021032/reports/report.html[0m
2026-11-14 02:15:02 - evalscope - [32mINFO[0m: Finished evaluation for Qwen3.8-27B on ['gsm8k'][0m
2026-11-14 02:15:02 - evalscope - [32mINFO[0m: Output directory: /workspace/meetup_qwen_20261114/results/gsm8k_bf16_mtp/20261114_021032[0m
EVAL_COMPLETED
```

#### Performance command and result

Save the two helper scripts in Appendix A before running this command. A new output tag preserves the original measured files:


```bash
bash /workspace/meetup_qwen_20261114/scripts/bench_128k.sh bf16_mtp_c4_reproduce 4
```


Original final output from `logs/bf16_mtp_c4.log`:


```text
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 4         
Successful requests:                     16        
Benchmark duration (s):                  113.00    
Total input tokens:                      2048000   
Total input text tokens:                 2048000   
Total generated tokens:                  16000     
Total generated tokens (retokenized):    12858     
Request throughput (req/s):              0.14      
Input token throughput (tok/s):          18124.48  
Output token throughput (tok/s):         141.60    
Peak output token throughput (tok/s):    357.00    
Peak concurrent requests:                6         
Total token throughput (tok/s):          18266.08  
Concurrency:                             3.84      
Accept length:                           3.51      
---------------------Goodput----------------------
SLOs (ms):                               tpot:50   
Request goodput (req/s):                 0.14      
SLO attainment (%):                      100.00    
TPOT SLO attainment (%):                 100.00    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   27117.24  
Median E2E Latency (ms):                 25799.24  
P90 E2E Latency (ms):                    38391.87  
P95 E2E Latency (ms):                    39615.16  
P99 E2E Latency (ms):                    42534.27  
---------------Time to First Token----------------
Mean TTFT (ms):                          5381.71   
Median TTFT (ms):                        2719.72   
P90 TTFT (ms):                           10531.69  
P95 TTFT (ms):                           11740.96  
P99 TTFT (ms):                           14611.58  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.76     
Median TPOT (ms):                        19.92     
P90 TPOT (ms):                           30.59     
P95 TPOT (ms):                           33.49     
P99 TPOT (ms):                           33.97     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           21.76     
Median ITL (ms):                         10.71     
P90 ITL (ms):                            14.48     
P95 ITL (ms):                            21.72     
P99 ITL (ms):                            76.67     
Max ITL (ms):                            6732.15   
----------------Cache Hit Details-----------------
Total prompt tokens:                     2048000   
Total cached tokens:                     1843200   
  Device:                                1843200   
  Host:                                  0         
Cache hit rate:                          90.0%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
```

### Qwen3.8-27B-w8a8

The selected configuration enables MTP at concurrency 8: 32/32 requests satisfy TPOT <= 50 ms, mean TPOT 30.07 ms, P99 TPOT 48.23 ms, output throughput 215.59 token/s, and 90.00% cache hits.

#### Complete tuned launch command


```bash
#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset ASCEND_LAUNCH_BLOCKING
export NO_PROXY=localhost,127.0.0.1
export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export TASK_QUEUE_ENABLE=1
export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
exec python3 -m sglang.launch_server \
 --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
 --host 127.0.0.1 --port 6688 --tp-size 4 --nnodes 1 \
 --attention-backend ascend --device npu \
 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
 --mamba-radix-cache-strategy extra_buffer \
 --max-running-requests 16 --max-mamba-cache-size 128 \
 --mem-fraction-static 0.80 --context-length 131072 \
 --cuda-graph-bs-decode 1 2 4 8 12 16 \
 --quantization modelslim --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

#### Accuracy command

Run after the model server is ready:


```bash
export NO_PROXY=localhost,127.0.0.1
evalscope eval \
 --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
 --api-url http://127.0.0.1:6688/v1 --api-key EMPTY \
 --eval-type openai_api --datasets gsm8k --limit 200 --eval-batch-size 16 --seed 42 \
 --generation-config '{"temperature":0,"max_tokens":8192,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
 --work-dir /workspace/meetup_qwen_20261114/results/gsm8k_w8a8_mtp
```


Result: **197/200 (98.50%)**, 200 succeeded, 0 errors. The non-MTP baseline scored 196/200 (98.00%). Original final EvalScope output follows; full stdout and per-example artifacts are in the attached evidence archive.


```text
Overall report table: 
┌──────────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model            │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├──────────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98.5%   │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘ 
[0m
2026-11-14 03:03:43 - evalscope - [32mINFO[0m: Overall perf table: 
Model             Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
----------------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
Qwen3.8-27B-w8a8  GSM8K        200  8.571 s    —           —           41.18 tok/s       689        353 
[0m
2026-11-14 03:03:44 - evalscope - [32mINFO[0m: HTML report generated: /workspace/meetup_qwen_20261114/results/gsm8k_w8a8_mtp/20261114_030048/reports/report.html[0m
2026-11-14 03:03:44 - evalscope - [32mINFO[0m: Finished evaluation for Qwen3.8-27B-w8a8 on ['gsm8k'][0m
2026-11-14 03:03:44 - evalscope - [32mINFO[0m: Output directory: /workspace/meetup_qwen_20261114/results/gsm8k_w8a8_mtp/20261114_030048[0m
EVAL_COMPLETED
```

#### Performance command and result

Save the two helper scripts in Appendix A before running this command. A new output tag preserves the original measured files:


```bash
bash /workspace/meetup_qwen_20261114/scripts/bench_128k.sh w8a8_mtp_c8_reproduce 8
```


Original final output from `logs/w8a8_mtp_c8.log`:


```text
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 8         
Successful requests:                     32        
Benchmark duration (s):                  148.43    
Total input tokens:                      4096000   
Total input text tokens:                 4096000   
Total generated tokens:                  32000     
Total generated tokens (retokenized):    29256     
Request throughput (req/s):              0.22      
Input token throughput (tok/s):          27595.13  
Output token throughput (tok/s):         215.59    
Peak output token throughput (tok/s):    541.00    
Peak concurrent requests:                10        
Total token throughput (tok/s):          27810.72  
Concurrency:                             7.83      
Accept length:                           3.50      
---------------------Goodput----------------------
SLOs (ms):                               tpot:50   
Request goodput (req/s):                 0.22      
SLO attainment (%):                      100.00    
TPOT SLO attainment (%):                 100.00    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   36321.58  
Median E2E Latency (ms):                 35148.87  
P90 E2E Latency (ms):                    49334.06  
P95 E2E Latency (ms):                    55302.59  
P99 E2E Latency (ms):                    59802.10  
---------------Time to First Token----------------
Mean TTFT (ms):                          6280.88   
Median TTFT (ms):                        2481.39   
P90 TTFT (ms):                           19023.60  
P95 TTFT (ms):                           19156.71  
P99 TTFT (ms):                           19172.00  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.07     
Median TPOT (ms):                        29.82     
P90 TPOT (ms):                           41.59     
P95 TPOT (ms):                           44.19     
P99 TPOT (ms):                           48.23     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           30.08     
Median ITL (ms):                         14.35     
P90 ITL (ms):                            25.24     
P95 ITL (ms):                            38.97     
P99 ITL (ms):                            567.83    
Max ITL (ms):                            10807.36  
----------------Cache Hit Details-----------------
Total prompt tokens:                     4096000   
Total cached tokens:                     3686400   
  Device:                                3686400   
  Host:                                  0         
Cache hit rate:                          90.0%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
```

### Parameter changes and concurrency search

Both baselines used TP=4, `--chunked-prefill-size 32768`, `--max-prefill-tokens 32768`, `--mamba-radix-cache-strategy extra_buffer`, `--max-running-requests 16`, `--max-mamba-cache-size 128`, `--mem-fraction-static 0.80`, context length 131072, and decode graph batch sizes 1/2/4/8/12/16. The W8A8 path additionally used `--quantization modelslim`.

The change from each baseline to MTP added these environment variables and arguments:

```bash
export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
# Additional launch arguments:
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

All other launch settings remained the same within each model's comparison. To reproduce a baseline, remove these three exports and four speculative-decoding arguments from its complete tuned launch script. The evidence archive also contains all four original launch scripts.

Benchmark concurrency varied the offered load; `--max-running-requests` remained 16. BF16 baseline concurrency was 1/2/4/8/16 and MTP concurrency was 4/8/12/16. W8A8 baseline concurrency was 4/8 and MTP concurrency was 4/8/12. No W8A8 concurrency-16 result is claimed.

### Accuracy comparison

| Model | Baseline | MTP | Successful / errors per evaluation |
| --- | ---: | ---: | --- |
| Qwen3.8-27B | 195/200 (97.50%) | 196/200 (98.00%) | 200 / 0 |
| Qwen3.8-27B-w8a8 | 196/200 (98.00%) | 197/200 (98.50%) | 200 / 0 |

The tuned scores differ by one example (0.5 percentage points). This small, fixed-sample comparison does not demonstrate generally higher accuracy from quantization or MTP.

### Best observed performance under each TPOT criterion

| Criterion | Model / mode | Concurrency | Output token/s | Mean TPOT ms | P99 TPOT ms | Requests <= 50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Every measured request | BF16 MTP | 4 | 141.60 | 21.76 | 33.97 | 100% |
| Every measured request | W8A8 MTP | 8 | 215.59 | 30.07 | 48.23 | 100% |
| Mean only | BF16 MTP | 12 | 233.48 | 39.93 | 64.51 | 75.00% |
| Mean only | W8A8 MTP | 12 | 241.70 | 40.00 | 63.88 | 79.17% |

Under the every-request criterion, the selected W8A8 configuration has 52.25% higher observed output throughput, with a higher selected concurrency (8 versus 4). At the same concurrency of 12 under the mean-only criterion, W8A8 is 3.52% higher. These are single-run observations and do not establish a stable small performance advantage.

At concurrency 4, enabling MTP changes BF16 output throughput from 100.28 to 141.60 token/s and W8A8 from 90.39 to 163.22 token/s. All four concurrency-4 runs meet the every-request TPOT limit. The BF16 MTP concurrency-16 run reaches mean TPOT 50.80 ms and is excluded from the mean-compliant optimum.

Weight-loading log `mem usage` is 12.79 GB per device for BF16 and 7.42 GB for W8A8, about 42% lower. These values describe weight loading, not total service HBM allocation; caches and execution graphs allocate additional memory.

## 3. Reference documentation feedback

These are subjective usability scores for this task, not independent quality measurements.

| Area | Score (1-5) | Observation |
| --- | ---: | --- |
| Installation and deployment | 4 | The provided environment and model paths were usable; Qwen3.8 required adapting the Qwen3.6 recipe. |
| Best practices | 4 | Mamba, MTP, and ModelSlim settings were useful; the example needed adaptation to four A2 devices. |
| Parameter tuning | 4 | Baseline, concurrency, memory, and post-MTP accuracy checks supported a structured experiment. |
| Accuracy evaluation | 4 | EvalScope was usable; an explicit thinking mode, few-shot count, and generation limit improve comparisons. |
| Performance evaluation | 3 | Exact prefix-only warmup and verification of actual cache hits should accompany shared-prefix length settings. |

## 4. Observed issues and suggested improvements

1. The container's OpenAI SDK 2.6.1 lacks `ResponseFunctionShellToolCall`, so optional `/v1/responses` initialization reports an import failure. Chat-completions evaluation remained usable. A separate dependency PR changes the upstream NPU manifest `python/pyproject_npu.toml` from 2.6.1 to 2.8.0, matching the generic manifest. An isolated 2.8.0 installation was verified to import `OpenAIServingResponses`; the model experiments in this report predate that isolated check and were not rerun with the upgraded SDK.
2. Configuring a 90% shared prefix does not itself guarantee 90% measured cache hits. Document the sequence: clear cache, warm only the shared prefix, then measure without another flush. Record token lengths and actual cache-hit counts alongside throughput.
3. A shared Qwen3.8 / four-card A2 example would reduce adaptation effort. State whether the 50 ms TPOT requirement is a mean, percentile, or every-request condition. Record the tested version and workload before interpreting a performance change.

This PR contributes the experiment record and reproducible commands. It does not change runtime code, claim a new benchmark framework, or claim that the task's administrative acceptance and submission requirements are already complete. The official task number and developer ID remain pending.

## Appendix A. Complete performance scripts

Create `/workspace/meetup_qwen_20261114/scripts`, `logs`, and `results` if needed. Save the following scripts at the named paths. The scripts are preserved from the experiment; they rely on the recorded SGLang benchmark implementation and local tokenizer files.

### `scripts/bench_128k.sh`


```bash
#!/usr/bin/env bash
set -euo pipefail
TAG=$1
CONCURRENCY=$2
PROMPTS=$((CONCURRENCY * 4))
BASE=/workspace/meetup_qwen_20261114
export NO_PROXY=localhost,127.0.0.1
unset SGLANG_IS_IN_CI
python "$BASE/scripts/warm_prefix.py" "$PROMPTS" | tee "$BASE/logs/$TAG.warmup.log"
python -m sglang.benchmark.serving --dataset-name generated-shared-prefix --backend sglang --host 127.0.0.1 --port 6688 --gsp-num-groups 1 --gsp-prompts-per-group "$PROMPTS" --gsp-system-prompt-len 115200 --gsp-question-len 12800 --gsp-output-len 1000 --gsp-range-ratio 1 --tokenize-prompt --max-concurrency "$CONCURRENCY" --num-prompts "$PROMPTS" --request-rate inf --seed 42 --temperature 0 --warmup-requests 0 --cache-report --goodput tpot:50 --output-details --output-file "$BASE/results/$TAG.jsonl" 2>&1 | tee "$BASE/logs/$TAG.log"
```


### `scripts/warm_prefix.py`


```python
import json,random,sys,hashlib
import numpy as np
import requests
from transformers import AutoTokenizer
from sglang.benchmark.datasets.generated_shared_prefix import sample_generated_shared_prefix_requests
n=int(sys.argv[1])
random.seed(42);np.random.seed(42)
s=requests.Session();s.trust_env=False
base='http://127.0.0.1:6688'
r=s.get(base+'/v1/models',timeout=30);r.raise_for_status()
model=r.json()['data'][0]['id']
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
rows=sample_generated_shared_prefix_requests(num_groups=1,prompts_per_group=n,system_prompt_len=115200,question_len=12800,output_len=1000,range_ratio=1,tokenizer=tokenizer,seed=42,input_ids=True)
prefix=list(rows[0].prompt[:115200])
assert len(prefix)==115200 and all(len(x.prompt)==128000 and list(x.prompt[:115200])==prefix for x in rows)
r=s.post(base+'/flush_cache',timeout=120);r.raise_for_status()
r=s.post(base+'/generate',json={'input_ids':prefix,'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True}},timeout=600);r.raise_for_status()
result=r.json()
print(json.dumps({'phase':'shared_prefix_warmup','model':model,'measured_requests':n,'prefix_tokens':len(prefix),'prefix_sha256':hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),'meta_info':result.get('meta_info')},ensure_ascii=False))
```


## Appendix B. All recorded performance runs

Throughput is **output** tokens per second, not input-plus-output throughput. Every valid row has exact 128,000/1,000 input/output lengths and zero errors. The excluded diagnostic is shown to explain the cache-protocol correction.

| Run tag | C | Completed | Cache % | Mean TPOT ms | P99 TPOT ms | Output token/s | Requests <= 50 ms | Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16_base90_c1 | 1 | 4 | 90.00 | 22.03 | 22.04 | 40.70 | 100.00% | Every request passes |
| bf16_base90_c2 | 2 | 8 | 90.00 | 26.08 | 27.16 | 66.58 | 100.00% | Every request passes |
| bf16_base90_c4 | 4 | 16 | 90.00 | 31.85 | 37.02 | 100.28 | 100.00% | Every request passes |
| bf16_base90_c8 | 8 | 32 | 90.00 | 44.23 | 56.20 | 135.46 | 84.38% | Mean only passes |
| bf16_base90_c16 | 16 | 64 | 90.00 | 70.81 | 93.52 | 166.20 | 0.00% | Mean exceeds limit |
| bf16_baseline_c1 | 1 | 4 | 64.20 | 22.03 | 22.04 | 32.47 | 100.00% | Excluded: cold-cache diagnostic |
| bf16_mtp_c4 | 4 | 16 | 90.00 | 21.76 | 33.97 | 141.60 | 100.00% | Every request passes |
| bf16_mtp_c8 | 8 | 32 | 90.00 | 34.27 | 59.93 | 188.46 | 90.62% | Mean only passes |
| bf16_mtp_c12 | 12 | 48 | 90.00 | 39.93 | 64.51 | 233.48 | 75.00% | Mean only passes |
| bf16_mtp_c16 | 16 | 64 | 90.00 | 50.80 | 80.14 | 248.13 | 50.00% | Mean exceeds limit |
| w8a8_baseline_c4 | 4 | 16 | 90.00 | 36.95 | 41.59 | 90.39 | 100.00% | Every request passes |
| w8a8_baseline_c8 | 8 | 32 | 90.00 | 48.58 | 59.65 | 128.27 | 75.00% | Mean only passes |
| w8a8_mtp_c4 | 4 | 16 | 90.00 | 19.13 | 26.13 | 163.22 | 100.00% | Every request passes |
| w8a8_mtp_c8 | 8 | 32 | 90.00 | 30.07 | 48.23 | 215.59 | 100.00% | Every request passes |
| w8a8_mtp_c12 | 12 | 48 | 90.00 | 40.00 | 63.88 | 241.70 | 79.17% | Mean only passes |


## Appendix C. Evidence and references

The PR description attaches original browser screenshots and `qwen38-meetup-experiment-evidence.zip`. The archive contains the original environment output, all launch/evaluation/benchmark scripts, full stdout logs, per-example GSM8K predictions/reviews and reports, raw performance JSONL, summary CSVs, metadata, and SHA-256 checksums. No model weights are included. Source paths such as `logs/bf16_mtp_c4.log` refer to paths inside that archive.

The original experiment logs and measured values are retained without redaction. The report's result blocks are the final tool summaries; the archive retains progress output and other diagnostic lines omitted from those blocks.

References used for this experiment:


- [Qwen3.6-27B best practices](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_6_27b)

- [NPU parameter tuning](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning)

- [Accuracy evaluation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation)

- [Performance testing](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/evaluation/performance_testing)
