# GLM-5.2 NVFP4 historical reproduction

These scripts preserve the request/tokenization protocols used for SGLang
[#41913](https://github.com/sgl-project/sglang/pull/41913)'s September 30, 2026
GB300 serving smoke and GSM8K evaluation. The measurements used SGLang
`7c547b1094`, FlashMLA integration pin
`f7d49d040c262f18b9973fe8cd5352669def43e8`, and
`lmsysorg/sglang:dev-cu13` (CUDA 13.0, PyTorch 2.13.0+cu130,
Transformers 5.12.1, FlashInfer 0.6.18). The image tag is mutable; these
versions describe the historical environment, not a pinned container digest.

**The throughput numbers were not produced by `sglang.bench_serving`.** They
came from the custom `pr41913_serving_probe_v2.py` client, reproduced here as
`serving_probe.py` with configurable tokenizer and output paths. The original
GSM8K client is reproduced as `eval_gsm8k.py`; the generic few-shot evaluator
is not an exact replacement for this protocol.

## Server

Use four GB300 GPUs and a local `nvidia/GLM-5.2-NVFP4` checkpoint. Both cases
use the same model weights and change only the KV-cache dtype. The historical
container mounted the checkpoint at `/model`; its physical mount/revision
was not preserved in the recovered launcher artifacts.

Run each case separately (stop the previous server before starting another):

```bash
export MODEL_PATH=/path/to/GLM-5.2-NVFP4
export KV_DTYPE=fp8_e4m3  # repeat with nvfp4
export PORT=37000        # historical NVFP4 run used 35000

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp-size 4 --dp-size 4 --enable-dp-attention \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --kv-cache-dtype "$KV_DTYPE" --page-size 64 \
  --max-running-requests 32 --mem-fraction-static 0.80 \
  --chunked-prefill-size 4096 --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 32 --load-format runai_streamer \
  --host 127.0.0.1 --port "$PORT"
```

Historical resolved server arguments show `chunked_prefill_size=1024`, despite
4096 being requested. The checkpoint resolved to `modelopt_fp4` quantization
and `flashinfer_trtllm` MoE; neither was explicitly selected in the launch
script. Decode used CUDA graphs, prefill was eager, and radix caching was
left enabled. RunAI was used to avoid a weight-loading stall.

For the historical EAGLE/MTP variant, add:

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 5 --speculative-eagle-topk 1 \
--speculative-num-draft-tokens 6
```

## Serving smoke

Run the client after the server is ready:

```bash
python3 benchmark/glm52_nvfp4/serving_probe.py \
  --model-path "$MODEL_PATH" --tag fp8_graph_final \
  --port 37000 --output-dir results

# Against the separately launched NVFP4 server:
python3 benchmark/glm52_nvfp4/serving_probe.py \
  --model-path "$MODEL_PATH" --tag nvfp4_graph_final \
  --port 35000 --output-dir results
```

The client first executes the historical short smokes and prefix-reuse
requests (2047, 2048, 8192, and 32768 tokens; the `nvfp4_graph*` tag also
executes the 200000-token prefix smoke). Then it sends concurrent batches
of 1/4/16/32 requests, each with 2048 input tokens and a maximum of 128 output
tokens, temperature 0. There is no Poisson arrival rate: each batch is
submitted together through a thread pool. There are 3 repetitions per
concurrency, repeat 0 is warmup, and the reported rate is the median of repeats
1 and 2. Tokens are counted from actual `completion_tokens`, so EOS can end a
request early. Distinct leading prefixes reduce reuse, but cache is not
flushed and shared suffixes remain; this is a serving smoke, not an isolated
cache-format benchmark.

| Concurrency | Historical FP8 tokens/s | Historical NVFP4 tokens/s |
| ---: | ---: | ---: |
| 1 | 53.8 | 54.3 |
| 4 | 231.3 | 231.2 |
| 16 | 537.4 | 534.2 |
| 32 | 545.2 | 556.9 |

These historical runs used H128 prefill for FP8 but H64 prefill for NVFP4.
The NVFP4-only H64 prefill selection has since been reverted: current
Blackwell routing uses H128 for both formats. The table must not be presented
as a measurement of the current head or a cache-format-only comparison.

## GSM8K

Run after the serving probe, as in the historical server script:

```bash
python3 benchmark/glm52_nvfp4/eval_gsm8k.py \
  --port 37000 --tag fp8_graph_final --questions 500 --output-dir results

# Against the separately launched NVFP4 server:
python3 benchmark/glm52_nvfp4/eval_gsm8k.py \
  --port 35000 --tag nvfp4_graph_final --questions 500 --output-dir results
```

The first 500 test questions are evaluated with the first 20 **training**
examples as raw-completion few-shot prompts, temperature 0, max 512 new
tokens, concurrency 32, and stops `Question`, `Assistant:`, `<|separator|>`.
The scorer extracts the last unsigned integer after removing commas. The
script checks the public GSM8K JSONL checksums against the recovered dataset
files rather than silently accepting changed data.

## Evidence limits

The historical `Initialized DSA NVFP4 global scales for 78/78 local layers`
message does **not** prove calibrated checkpoint scales: initialization also
counts fallback values of `1.0`. Check the checkpoint's attention scale
weights and loaded per-layer scales before making a calibration claim.
The available JET artifact at revision
`aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` has **no serialized calibrated
attention K/V scales**: its safetensors index contains 232385 tensors across
47 shards, with all 172800 scale tensors belonging to MLPs. Alternative
KV/projection scale names and `amax` tensors are also absent. Its export
quantization summary lists 156 K/V BMM quantizers, all `amax=dynamic`.
The historical loader would use fallback K/V scales of `1.0` for this
artifact, not calibrated per-layer K scales.

That artifact's config matches the recovered historical config byte for
byte (SHA256 `d3783a603e5aa9cb58eff7a5d8ac9c42d83156b229efe185c72e9e2dc8444923`).
This does **not** establish the historical weight revision or observed
runtime scale values. The old Slurm job is no longer queryable, and the
physical `/model` mount remains unconfirmed; therefore the historical
accuracy results must not be described as calibrated-K-scale validation.

Historical raw logs and outputs are retained in
`/dockerroot/containers/c/pr41913-gpu-validation/pr41913_validation_evidence.tar.gz`
and checkpoint metadata evidence in `/tmp/pr41913-checkpoint-evidence/`
in the author's validation environment; neither is bundled with this repo.
