# Target-only offload and a separate MTP tail

## Draft KV policy

The process-wide Ascend sparse-offload environment flag applies to the target
model only. Draft attention backends must not create or inherit a sparse KV
manager. Their native MLA pool retains CKV, KPE and indexer KV in HBM.
The pool configurator charges the full draft MLA layout even when the target
has an index-only variable pool; it does not reserve a draft host-KV manager.

Allocation logs expose the role, offload policy and CKV/KPE byte counts.
In the tested BF16, one-layer draft with98304 native token slots, each rank
allocated100794368 CKV bytes and12599296 KPE bytes. A stable profile showed
native LightningIndexer/SparseFlashAttention in the draft graphs and no draft
CopySFA tasks.

## Optional prefix-plus-tail selection

`SGLANG_NPU_MTP_SEPARATE_KV_TAIL=1` enables a separate target HBM tail for the
linear NEXTN3/1/4 path. This is an explicit selection contract, not a silent
replacement for the existing exact2048 full-visible selection.

- LIM selects2048 historical KV entries before the page-aligned start of the
  current four-row candidate window. These entries alone occupy the C pool.
- Two128-token pages outside C retain the recent suffix. Every layer writes
  current KV directly into these HBM pages. Missing stable suffix rows are
  restored through the canonical native physical host-page mapping.
- CopySFA attends to the selected historical entries plus the causally visible
  suffix. Rotated HBM page tables agree with the circular physical write slots.
- Metadata keeps graph addresses stable. Rewritten candidates overwrite their
  local suffix slots; request reuse invalidates the suffix tags. Full history
  is still written to DRAM for later retrieval after suffix pages rotate out.
- The generic full-visible resident-refresh path remains available with the
  new flag disabled. Both metadata fusions are supported.

Native payload movement uses the existing `unidex_copy_inplace` operator.
Only the small index/tag preparation is implemented in Triton; a prototype
Triton payload mover was discarded after compilation/runtime failures.
The existing paired CopySFA kernel package is sufficient; no new C++ ABI is
introduced by this framework follow-up.

The additional tail bytes and graph-stable workspaces are included in static
memory accounting. C remains a128-token-aligned budget independent of the
two extra suffix pages. Numerical tests sample capacities11392,12288 and16384;
these samples do not establish exhaustive coverage of every capacity.

## Validation

- Four CPU policy tests check target enablement, draft bypass and native draft
  cell pricing.
- Twenty existing CPU cache/lifecycle regressions pass with the optional mode
  disabled. The standalone manager fixture explicitly initializes that mode.
- Five real-manager tests pass with the suffix mode enabled: eager/Graph,
  fused/unfused request metadata, BF16/H4, C11392/C12288/C16384, cold fill, rewritten
  candidates, request reuse, padding and rollback. Attention is checked against
  the selected-prefix-plus-causal-tail mathematical reference.
- An existing full-visible real-manager Graph test also passes with the new
  suffix flag disabled, including rewritten residents and rollback.

The serving comparison uses A3 TP16/DP1, GLM-5.2 W4A8/BF16 KV, DeepEP auto,
NEXTN3/1/4, graph bucket16, C12288, shared64000-token prompts and1024 outputs,
N16/C16. Timed runs and stable20-step profiles are separate. The article's
table9 CopySFA24.01ms row sums78 target layers per step. TPOT/TPS derived from
a profile assume three output tokens per step; they are not client metrics.

Prefix-cache hits do not establish the selected-KV C-pool hit rate. The
serving workload and operator boundaries are aligned to the article, while
identical compiler/runtime binaries and identical historical miss traces
have not been established. Broad model accuracy remains outside this check.

## Observed serving results

With the draft policy repaired and the suffix flag disabled, all16 requests
completed1024 outputs. Client output throughput was129.576 tokens/s, mean
TPOT80.013ms and TTFT24244.775ms. Cached-prefill admission caused long streaming
gaps. The separate stable profile had19 complete steps, each with21 LIM and78
CopySFA tasks: CopySFA30.227ms, target envelope106.275ms and full step113.106ms.
The article's three-token assumption gives TPOT37.702ms and424.379 tokens/s.

The optional prefix-plus-tail run also completed16x1024 outputs. Its client
throughput was188.674 tokens/s, TPOT61.041ms and TTFT19597.422ms. However its
stable profile was slower: CopySFA36.811ms, target envelope122.057ms and full
step129.333ms, or43.111ms /371.135 tokens/s under the article's assumption.
Each of its19 complete steps also contained exactly21 LIM and78 CopySFA tasks.

Client average acceptance lengths were2.942 and2.793 respectively, including
bonus tokens. Each request directly reported63872/64000 cached prompt tokens.
These are two separate fresh-process observations, not a same-address causal
operator A/B. The optional tail scheme changes the selected attention set.
It is therefore kept disabled by default: the higher client throughput is not
evidence of faster steady decode. The article's reference is24.01ms CopySFA,
109.54ms per step and36.51ms /438.2 tokens/s assuming three tokens per step.

The optional-tail service additionally passed a seven-request distant-answer
check at64000 input tokens: cold1, hot2 and hot4 all returned the expected
eight-digit identifier. Hot requests directly reported63872 cached tokens;
all seven reported acceptance length3.0. This is a bounded retrieval check,
not a broad model-accuracy evaluation.
