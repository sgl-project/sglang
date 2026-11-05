# PD decode input-only prefix cache evidence

This draft adds an opt-in retention policy, not a new speculative decoding
algorithm. Current upstream already accepts speculative decoding with decode
radix caching. With this option, only the page-aligned input prefix enters the
decode tree. Generated output remains request-owned and is freed by the normal
release path. Default behavior is unchanged.

## Historical hardware evidence

`historical-correctness.json` contains sanitized observations from a 2026-10-28
GLM-5.2 W4A8C8 deployment on four Ascend 910B4 nodes (two prefill and two decode,
32 NPUs), CANN 9.1.0, NEXTN 5/1/6 on decode. That deployment used the older,
model-specific `SGLANG_GLM52_D_INPUT_CACHE` patch and additional deployment
changes. It was not this upstream revision.

Three synthetic records use approximately 2k, 8k and 16k input tokens. Each is
queried cold, repeated, queried after flushing the caches, and followed by a
conversation continuation: 12 requests per arm. Temperature is zero, top-p is
one, maximum output is 96 tokens, and thinking is disabled. The outputs match
the no-decode-prefix-cache control and the expected record in all 12 cases.
Independent decode logs show prefix hits of 2,048, 8,448 and 16,896 tokens.

The fixture puts a unique code and integer before repeated unrelated log lines,
asks for those values as JSON, then asks to keep the code and add seven to the
integer. The output records and hashes of the original observation files are
included. No private endpoints, model paths, credentials or user prompts are
published.

## Current draft

Set these variables on the decode service and enable
`--disaggregation-decode-enable-radix-cache`:

```bash
export SGLANG_PD_DECODE_CACHE_INPUT_ONLY=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
```

The first port requires EAGLE/NEXTN, Mooncake, PP/CP/DCP=1 and one FULL device
cache component. It rejects HiCache on decode, decode KV offload, external KV
stores, session radix tracking and unsupported tree configurations. A separate
prefill process can continue to use HiCache. There are no model-layer-count or
draft-step hard-coded checks in this port; broader hardware support is **not**
claimed by removing those historical checks.

Run the CPU regression without a server or model weights:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_pd_decode_input_only_cache.py
```

It exercises the real Python tree, request table and release path, including
EAGLE bigram boundaries at 127/128/129 and 255/256/257 input tokens, duplicate
insertion, a changed input suffix, in-flight prefix locks, output-tail lifetime,
release without insertion and default-off/prefill behavior. The CPU paged allocator
uses physical 128-token allocation units; this is not a test of the NPU backend,
PD transfer or target/draft KV contents.

## Required before requesting merge

- Run this exact upstream revision and the TDP backport on the four-node A2
  GLM-5.2 W4A8C8 topology, including their required INT8 KV enablement.
- Compare cache-hit and recomputation outputs/logits with fixed inputs; validate
  both target and MTP draft KV at the retained boundary on real devices.
- Exercise concurrent shared prefixes, distinct suffixes and cache salts,
  eviction pressure, cancellation, timeout, retraction and repeated reuse. Check
  pool accounting and lock balance as well as responses.
- Evaluate non-greedy sampling and long-context quality/acceptance behavior.
- Measure isolated same-workload TTFT, per-request Decode and OutputTPS with
  independent repeats. Existing full-deployment performance results do not
  establish this patch's isolated benefit.

The latest-main port and TDP backport have **no new hardware validation yet**.
This evidence supports draft review only; no zero-regression or standalone speedup
claim is made.
