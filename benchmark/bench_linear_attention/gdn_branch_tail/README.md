# Hybrid Linear branch and tail checkpoints: GDN implementation

Hybrid prefix reuse requires a recoverable Linear state at the cached boundary,
including the convolution history and recurrent state at the same token depth.
When an extend tracks only one boundary, selecting an interior branch can leave
its computed aligned prompt tail without a separate checkpoint. Retaining both
requires independent state ownership for the two boundaries.

This patch implements that pattern for GDN: it captures the branch into one
spare state slot while native tracking captures the aligned tail in the same
forward, without splitting the extend. See the [backend evidence index](backend-evidence.md)
for the mechanism comparison and the separately validated KDA prototype.

Enable it with `SGLANG_EXPERIMENTAL_GDN_BRANCH_AND_TAIL=1`. It defaults to
disabled. The first supported configuration uses the
`Qwen3_5ForConditionalGeneration` architecture, CUDA Triton GDN, BF16 recurrent
state, `extra_buffer`, eager execution, and unified serving. TP1 and TP2 are
the validation targets.

The scheduler chooses at most one extra snapshot per batch, in request order.
Other candidates use native branch tracking. Each TP rank must have sufficient
spare capacity after reserving native checkpoint headroom for the batch.
A rank with insufficient capacity makes every rank use native tracking.

The feature applies only to the final Prefill extend. PD, HiCache, overlap,
speculative decoding, prefill CUDA graphs, other state dtypes and other backends
use the native path. The initial architecture restriction bounds the supported
conv and recurrent state layout.

Native checkpoint publication and path-cap eviction still apply. In unified
Level 2 serving a tracked prompt tail can remain request-owned during Decode.
A later native Decode checkpoint may make both modes resume at the same depth.

## Reproduction

Use the same model, dependencies, device, page size and actual pool capacities
for each arm. With a CUDA SGLang environment installed, launch an arm with the
environment variable set to `0` or `1`.

```bash
SGLANG_EXPERIMENTAL_GDN_BRANCH_AND_TAIL=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B --language-only --dtype bfloat16 \
  --tp-size 1 --attention-backend triton \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
  --page-size 64 --hybrid-cache-checkpoints-level 2 --mamba-max-states-per-path 1 \
  --max-mamba-cache-size 64 --max-total-tokens 32768 \
  --context-length 8192 --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --disable-overlap-schedule --disable-prefill-cuda-graph --disable-decode-cuda-graph
```

The registered end-to-end test independently rebuilds the tree for the branch
and continuation probes. A 6,049-token producer executes as 4,096 + 1,953 tokens.
The branch probe resumes at 5,952 in both modes. With one or 32 generated tokens,
the input-only continuation resumes at 5,952 with native tracking and at 6,016
with the optional capture. The test also covers 128-token Decode and records
teacher-forced suffix logprobs against cold and native-restored controls.

```bash
python3 test/registered/unit/mem_cache/unified_radix_cache/test_gdn_branch_tail_cache.py
python3 test/registered/attention/test_gdn_branch_tail_states.py
QWEN35_GDN_TEST_MODEL=Qwen/Qwen3.5-4B GDN_BRANCH_TAIL_TEST_TP=1 \
  python3 test/registered/e2e/radix_cache/test_gdn_branch_tail_checkpoint.py
```

Set `GDN_BRANCH_TAIL_RESULT_PATH` to retain the end-to-end test's logprob
measurements. The cache tests use real allocators, components and tree operations.
With two visible CUDA devices they also exhaust one rank's pool and check
collective fallback. The layer test compares the complete recurrent and conv
snapshots with independent prefix forwards, and verifies that live states and
forward outputs are unchanged.

## Public trace measurements

The replay script consumes the pinned
[AgentX Weka corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729).
It reconstructs deterministic input IDs from per-session 64-token block hashes.
These reconstructed inputs preserve block-prefix relationships. They measure
cache and execution behavior with one generated token per request.

The directed selection contains four consecutive three-request sequences,
selected to expose a final-chunk branch followed by tail reuse. The unselected
selection takes eight eligible initial requests from each of twelve sessions
chosen by a fixed session-ID hash, without selecting for branch or cache benefit.

Use 65,536 context, 524,288 KV tokens and 128 state slots for these trace
measurements. Keep the remaining settings identical between arms. The replay
flushes warmup entries, uses a fixed sequential order, and records input
fingerprints, first streamed-token latency, actual cached tokens and outputs.
The directed run also includes four cold 6,049-token controls without candidates.

```bash
python3 benchmark/bench_linear_attention/gdn_branch_tail/replay.py \
  --base-url http://127.0.0.1:30000 --trace /path/to/weka.jsonl \
  --selection directed --output /path/to/results/directed
python3 benchmark/bench_linear_attention/gdn_branch_tail/replay.py \
  --base-url http://127.0.0.1:30000 --trace /path/to/weka.jsonl \
  --selection unselected --output /path/to/results/unselected
```

Repeat paired directed arms with alternating launch order. Report producer,
continuation and combined latency separately. The directed sample establishes
a conditional benefit. Coverage and overall service performance require their
own workload measurements.

The branch/tail placement problem and the two-checkpoint direction were
discussed in [SGLang #45379](https://github.com/sgl-project/sglang/pull/45379).
[SGLang #50059](https://github.com/sgl-project/sglang/pull/50059) supplies related
SWA publication and ownership regression examples.
