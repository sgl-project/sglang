## 1. Prepare docker image

### gfx942

```bash
docker run -it --name mimo-v26-flash-opt-gfx942 --ipc=host --network=host --privileged --security-opt seccomp=unconfined \
--cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  \
-v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510
```

## 2. Environment setup

### Clean up pre-installed environment

```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang

```bash
cd /sgl-workspace
git clone https://github.com/sammysun0711/sglang -b mimo-opt-mxfp4-dflash
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```

### Install AITER & pyhip dependency

```bash
git clone https://github.com/sammysun0711/aiter -b integration/mimo-fp4-dflash
cd aiter
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL

Install FlyDSL runtime version that compatible with aiter

```bash
pip install flydsl==0.3.2
```

### Install MORI

For MORI expert parallelism, replace the Docker image's pre-installed MORI
1.1.1 with the 1.2.4 release wheel.

```bash
pip uninstall -y amd_mori
rm -rf /sgl-workspace/mori
pip install amd_mori==1.2.4
```

## 3. Run DP2/TP8 with dp attention baseline single node prefill benchmark

The main baseline follows the current launchers: FlyPA paged prefill enabled, FlyDSL target/draft attention, and BF16 KV cache.

### Launch TBO-off server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
CHUNKED_PREFILL_SIZE=32768 \
MEM_FRACTION_STATIC=0.75 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=0 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
SGLANG_AITER_PA_DECODE_IMPL=flydsl \
./launch_mimo_v26_flash_dp2_tp8_ep1_dpa_accuracy_baseline.sh
```

### Run TBO-off prefill benchmark

```bash
cd /sgl-workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```

### Run DP2 + attention TP4 + EP8 prefill benchmark

Launch global TP8 with DP attention, MORI EP8,
and DFlash block 8. TBO is disabled in this launcher. The global prefill chunk
is 32768, giving 16384 tokens per DP; the default MORI dispatch capacity is
also 16384 tokens per rank.

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
CHUNKED_PREFILL_SIZE=32768 \
MEM_FRACTION_STATIC=0.75 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=192 \
DISABLE_RADIX_CACHE=1 \
ENABLE_DFLASH=1 \
PERFORMANCE_MODE=1 \
./launch_mimo_v26_flash_dp2_tp4_ep8_dflash_accuracy_baseline.sh
```

`PERFORMANCE_MODE=1` enables fake expert routing. The server uses ordinary
serving, and the client requests one output token to measure prefill performance.

`SGLANG_MIMO_FUSED_RMS_QKV_QUANT=1` is the default: fused QKV RMSNorm/FP8
quantization packs activations and FP32 scales into one attention-TP all-gather.
Set it to `0` for the BF16 all-gather baseline. This requires the matching AITER
fused-quant wrapper with caller-owned output buffers.
In another terminal, run the full `mimo_v26_flash` preset:

```bash
cd /sgl-workspace/sglang/evaluation
BENCHMARK_PRESET=mimo_v26_flash \
HOST=127.0.0.1 \
LOG_DIR=./logs/mimo_v26_flash_dp2_attn_tp4_ep8_prefill \
./run_benchmark_mimo_pro_prefill.sh
```

The client keeps `TOKENIZE_PROMPT=0` by default and uses the preset's input
lengths, concurrency, and request counts. Each point runs four warmup requests
and flushes the cache before measurement. The client reports TTFT and input
token throughput for the whole server.

For **accuracy tests**, stop the performance server and rerun the launch command
with **`PERFORMANCE_MODE=0`** to enable real expert routing. Set `ENABLE_DFLASH=0`
for target-only accuracy or `ENABLE_DFLASH=1` for DFlash accuracy, then follow
the GSM8K evaluation steps in section 8.

## 4. Run DP2/TP8 with dp attention baseline single node decode benchmark with fake prefill

The standalone fake-prefill launcher defines the Flash baseline settings:

- TP4 x DP2, EP1, DFlash block 8, native MXFP4 weights with FlyDSL A8W4 decode kernel.
- FlyDSL attention, BF16 target/draft KV, and disabled Quick Reduce/mixed router.
- It adds `--disaggregation-mode decode --disaggregation-transfer-backend fake`.
- Acceptance is simulated at length 8 using `SGLANG_SIMULATE_ACC_LEN=3.5` and
  `SGLANG_SIMULATE_ACC_METHOD=match-expected`, Fake prefill skips prompt computation,
- so these runs measure performance; use the accuracy launcher with real prefill to evaluate accuracy.

The fake-prefill decode capacity must fit both the full-token and SWA-token
pools:

```text
(C + 1) * (ISL + OSL) <= token_capacity
(C + 1) * OSL         <= swa_layer_tokens
```

Apply these bounds per DP replica, using its admitted request count. Read each
worker's `token_capacity` from
`/server_info` -> `internal_states[].memory_usage.token_capacity`, and its
`swa_layer_tokens` from the server initialization log.

If SWA is the smaller bound for a short-ISL
case, increase `SWA_FULL_TOKENS_RATIO` only until the SWA slot count is at
least the full-token slot count (or the desired concurrency cap). Allocating
more than that reduces the full-token pool without increasing concurrency.

### Launch server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
MEM_FRACTION_STATIC=0.75 \
DISABLE_RADIX_CACHE=1 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
SGLANG_AITER_PA_DECODE_IMPL=flydsl \
./launch_mimo_v26_flash_dp2_tp8_ep1_dpa_decode_fake_prefill_baseline.sh
```

Use `DRY_RUN=1` to inspect the command without starting the server.

### Run the matching decode point

```bash
cd /sgl-workspace/sglang/evaluation
MODEL=/models/MiMo-V2.6-Flash-MOPD/ \
CASE_SPECS='65536|65536|64k|128' \
PROMPT_WAVES=4 \
WARMUP_REQUESTS=32 \
./run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

The client uses explicit token IDs, so no four-token subtraction is needed.
At the 1M context boundary, reserve additional slots for the scheduler:
use `CASE_SPECS='1047548|1047548|1m|10'` for 1,024 output tokens. An input
length of 1,047,552 leaves no scheduler margin and is capped at 1,022 output
tokens. This margin is separate from tokenizer special tokens.

### Run decode throughput analysis

```bash
python3 analyze_server_output_throughput.py \
  <path-to-server-log> \
  --dp-rank 0 \
  --target-bs <C-per-replica> \
  --exact-bs \
  --min-tps 40 \
  --max-tpot 25
```

Use the analyzer's filtered server median as the reported output throughput.
Repeat with `--dp-rank 1` for the other replica. Each scheduler reports its own
running requests and throughput: at balanced client concurrency 128, use
`--target-bs 64` per replica. Verify residency in the log before choosing this value. The analyzer requires explicit replica selection for logs containing multiple DP ranks. Report both medians; their sum is an aggregate estimate, not a time-aligned node-wide median.

The analyzer reports the following median-consistent derived values:

```text
TPS/request = filtered replica median / C-per-replica
TPOT (ms)   = 1000 * C-per-replica / filtered replica median
```

The analyzer also retains `mean_tpot_ms` and `mean_tps_per_request` as explicit
mean-based reference values.

<details>
<summary>5. Run DP2/TP4 w/o dp attention baseline single node prefill benchmark</summary>
The main baseline follows the current launchers: FlyPA paged prefill enabled, FlyDSL target/draft attention, and BF16 KV cache.

### Launch TBO-off server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
CHUNKED_PREFILL_SIZE=16384 \
MEM_FRACTION_STATIC=0.75 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=0 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
SGLANG_AITER_PA_DECODE_IMPL=flydsl \
./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_accuracy_baseline.sh
```

### Run TBO-off prefill benchmark

```bash
cd /sgl-workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```

MiMo-V2.6-Flash-MOPD DP2 x TP4 baseline defaults to `ENABLE_TWO_BATCH_OVERLAP=0`.
To opt into prefill-only TBO, stop the existing server and launch with:

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
CHUNKED_PREFILL_SIZE=16384 \
MEM_FRACTION_STATIC=0.75 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=1 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
SGLANG_AITER_PA_DECODE_IMPL=flydsl \
./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_accuracy_baseline.sh
```

### Run DP2/TP4 w/o dp attention TBO-on prefill benchmark

```bash
cd /sgl-workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```

Each DP replica overlaps its own TP4 collectives. DP attention remains disabled;
ordinary prefill can use TBO, while DFlash draft/verify and decode retain their
existing paths.

</details>

<details>
<summary> 6. Run DP2/TP4 w/o dp attention baseline single node decode benchmark with fake prefill</summary>

The standalone fake-prefill launcher defines the Flash baseline settings:
TP4 x DP2, EP1, DFlash block 8, load-time MXFP4-to-FP8 MoE conversion,
FlyDSL attention, BF16 target/draft KV, and disabled Quick Reduce/mixed router.
It adds `--disaggregation-mode decode --disaggregation-transfer-backend fake`.
Acceptance is simulated at length 4 using `SGLANG_SIMULATE_ACC_LEN=4` and
`SGLANG_SIMULATE_ACC_METHOD=match-expected`, matching the Pro fake-prefill
reference. This gives a controlled acceptance length for decode throughput
comparisons. Fake prefill skips prompt computation, so these runs measure
performance; use the accuracy launcher with real prefill to evaluate accuracy.

The fake-prefill decode capacity must fit both the full-token and SWA-token
pools:

```text
(C + 1) * (ISL + OSL) <= token_capacity
(C + 1) * OSL         <= swa_layer_tokens
```

Apply these bounds per DP replica, using its admitted request count. Read each
worker's `token_capacity` from
`/server_info` -> `internal_states[].memory_usage.token_capacity`, and its
`swa_layer_tokens` from the server initialization log.

If SWA is the smaller bound for a short-ISL
case, increase `SWA_FULL_TOKENS_RATIO` only until the SWA slot count is at
least the full-token slot count (or the desired concurrency cap). Allocating
more than that reduces the full-token pool without increasing concurrency.

### Launch server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
MEM_FRACTION_STATIC=0.75 \
DISABLE_RADIX_CACHE=1 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
SGLANG_AITER_PA_DECODE_IMPL=flydsl \
./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_decode_fake_prefill_baseline.sh
```

Use `DRY_RUN=1` to inspect the command without starting the server.

### Run the matching decode point

```bash
cd /sgl-workspace/sglang/evaluation
MODEL=/models/MiMo-V2.6-Flash-MOPD/ \
CASE_SPECS='65536|65536|64k|128' \
PROMPT_WAVES=4 \
WARMUP_REQUESTS=32 \
./run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

The client uses explicit token IDs, so no four-token subtraction is needed.
At the 1M context boundary, reserve additional slots for the scheduler:
use `CASE_SPECS='1047548|1047548|1m|10'` for 1,024 output tokens. An input
length of 1,047,552 leaves no scheduler margin and is capped at 1,022 output
tokens. This margin is separate from tokenizer special tokens.

### Run decode throughput analysis

```bash
python3 analyze_server_output_throughput.py \
  <path-to-server-log> \
  --dp-rank 0 \
  --target-bs <C-per-replica> \
  --exact-bs \
  --min-tps 40 \
  --max-tpot 25
```

Use the analyzer's filtered server median as the reported output throughput.
Repeat with `--dp-rank 1` for the other replica. Each scheduler reports its own
running requests and throughput: at balanced client concurrency 128, use
`--target-bs 64` per replica. Verify residency in the log before choosing this value. The analyzer requires explicit replica selection for logs containing multiple DP ranks. Report both medians; their sum is an aggregate estimate, not a time-aligned node-wide median.

The analyzer reports the following median-consistent derived values:

```text
TPS/request = filtered replica median / C-per-replica
TPOT (ms)   = 1000 * C-per-replica / filtered replica median
```

The analyzer also retains `mean_tpot_ms` and `mean_tps_per_request` as explicit mean-based reference values.

</details>

## 7. Profiling & Analysis

```bash
cd /sgl-workspace/sglang/evaluation
./run_sglang_profile.sh
```

## 8. Run GSM8K accuracy and acceptance gate

Use the real-prefill accuracy launcher with acceptance simulation disabled.
The runner uses upstream `/server_info` statistics and requires no SGLang
source changes. It rejects decode-only servers. Stop the fake-prefill server
and restart the real-prefill launcher from section 3 before this step.
The runner automatically installs EvalScope 1.11.0 in a separate client virtual
environment; no changes to the serving environment's dependencies are needed.

Run one command for the complete
GSM8K `main` test split, 4-shot prompting, and each worker's mean acceptance length >= 4:

```bash
cd /sgl-workspace/sglang/evaluation
MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/ \
MIN_ACCEPT_LENGTH=4.0 \
python3 ./run_evalscope_gsm8k_accuracy.py
```

This runs EvalScope once, then reads each worker's upstream
`avg_spec_accept_length` (including the bonus token). Every worker's mean
must meet `MIN_ACCEPT_LENGTH`; their arithmetic mean is also reported and
is explicitly unweighted. Raw counts for an exact global weighted mean are
not available from this API.

`FLUSH_CACHE=1` is required for the acceptance check. Use a dedicated server
with no concurrent inference or statistics resets during evaluation. The
upstream API does not expose simulation settings, so real acceptance must
be configured in the server launcher.

`MIN_GSM8K_ACCURACY=0.97` optionally enforces 97% accuracy;
`MIN_ACCEPT_LENGTH=4.0` sets the acceptance threshold for every worker.

## 9. Run swe-bench & accuracy benchmark test

Follow up customer's swe-bench accuracy verification guide.

## 10. H200 performance evaluation

Follow up customer's shared performance data
