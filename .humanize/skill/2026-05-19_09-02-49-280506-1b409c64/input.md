# Ask Codex Input

## Question

# Plan Critique Request — SGLang Double Sparsity Implementation

You are a planning Codex pass. The user is preparing an implementation plan for adding Double Sparsity to SGLang. Critique the draft, identify missing requirements, surface alternative directions, and propose candidate acceptance criteria. Do NOT just rephrase the draft — challenge assumptions.

## 1) Raw User Draft

```
I am in the middle of delivering a double sparsity implementation into SGLang.

Here are the immediate client requirements:
- Model: deepseek-ai/DeepSeek-V3.2 (FP8)
- Inference SLOs: 30 tokens/s with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, minimum concurrency: 16, Cache hit: ~55% (benchmark found in development/benchmark.sh)
- Page size: 64 (technically not explicitly listed as a hard requirement, but significantly preferred and implementation should support different page sizes)

Deferred Client requirements ordered from most important to least:
1. zai-org/GLM-5.
2. 128k ISL, 1024 OSL.
3. nvfp4 and mxfp4 quantizated weight support.

Downstream requirements after client deliverables:
1. Twilight (top-p selection instead of top-k)
2. Extensions as a general knob for the sglang engine
3. Integration into all other sglang features, like PD-Disagg and HiSparse.

Double Sparsity Implementation Sources (listed from most recent to least recent)
1. Twilight: https://github.com/tsinghua-ideal/Twilight, with https://github.com/tsinghua-ideal/flash-topk-attention/tree/d8803b29961c44d77a747636ad4282bd7a9094af
2. Legacy SGL implementation (parent commit of commit that removed double sparsity): https://github.com/sgl-project/sglang/tree/29f56cb2304bf6699da78e4e5a738fb794babcfd/python/sglang/srt/layers/attention
3. Original Author Implementation: https://github.com/andy-yang-1/DoubleSparse
4. Paper: https://arxiv.org/pdf/2408.07092

Hi-Sparse Implementation (Quite irrelevant, but an example of succesfull ship of performant sglang sparsity feature, design should ideally be inspired by this)
Guide: https://docs.sglang.io/docs/advanced_features/hisparse_guide
PRS: https://github.com/sgl-project/sglang/pull/20343, https://github.com/sgl-project/sglang/pull/23013, https://github.com/sgl-project/sglang/pull/21591

I am deciding between whether I should
1. Resume from the legacy sglang restoration: https://github.com/sgl-project/sglang/pull/22992
2. Resume from the current rewrite session: https://github.com/sgl-project/sglang/pull/25304/commits
3. Restart from scratch on a new branch of SGLang as this will be a huge downstream decision. I am leaning towards this as neither of the previous options were designed/created before the clients gave us these requirements

Help me first decide whether I should resume or restart from scratch on Sglang.
```

## 2) Pre-collected Repository Context (verified by reading current main)

The user's branch tree is at `/sgl-workspace/sglang` (main = 2a35707). Verified facts:

**Current main: Double Sparsity is fully removed.** Only `development/draft.md` and `development/plan.md` mention `double_sparsity` repo-wide. The legacy `python/sglang/srt/layers/attention/double_sparsity_backend.py` and `DoubleSparseTokenToKVPool` were removed in PR #23009 (the parent commit referenced in the draft).

**HiSparse framework already exists and is the model citizen for sparsity in SGLang.**
- Path: `python/sglang/srt/mem_cache/sparsity/`
- Subdirs: `algorithms/`, `backend/`, `core/`, `factory.py`
- Registered algorithms (factory.py): `quest`, `deepseek_nsa`
- Base class: `BaseSparseAlgorithm` with abstract `retrieve_topk()` and hooks `initialize_representation_pool`, `construct_representations`, `update_representations`. Base-class docstring lists ChunkKV, Quest, PQCache, SnapKV, Look-ahead QCache as expected algorithm slots — Double Sparsity is conceptually a peer.
- Backend adaptors: `NSABackendAdaptor`, `FlashAttentionAdaptor` (in `backend/backend_adaptor.py`).
- CLI flags: `--enable-hisparse`, `--hisparse-config` (JSON). `SparseConfig` already supports: `top_k`, `device_buffer_size`, `host_to_device_ratio`, `algorithm`, `backend`, `page_size`, `min_sparse_prompt_len`, `sparse_extra_config`.
- Validator (`arg_groups/hisparse_hook.py`): currently asserts `--enable-hisparse` only works for `is_deepseek_nsa` or `is_deepseek_v4` models, requires `--disable-radix-cache`, requires `kv_cache_dtype in {bfloat16, auto, fp8_e4m3}`, picks `flashmla_kv` for fp8_e4m3 and `flashmla_sparse` for bf16, and restricts `nsa_prefill_backend`/`nsa_decode_backend` accordingly.
- **`NSABackendAdaptor.adapt_for_attn_metadata` is a TODO stub** — meaning Quest-style algorithms cannot currently produce sparse attention on the MLA path. Only `deepseek_nsa` (which uses the model's own NSA indexer, not the framework selector) ships end-to-end on MLA today.

**DeepSeek V3.2 already has native sparse attention (NSA / DSA).**
- `python/sglang/srt/layers/attention/nsa/`: `nsa_indexer.py`, `dequant_k_cache.py`, `quant_k_cache.py`, Triton kernel, tilelang kernel, MTP precompute/verification.
- `python/sglang/srt/layers/attention/dsv4/`: V4 attention (compressor / indexer / metadata).
- The model has a learned index that selects ~top-k tokens; HiSparse on V3.2 is essentially routing memory tiering on top of this.

**Two open PRs by the user (Jiminator) target Double Sparsity:**

PR #22992 — `dev/double-sparsity-reintro` (created 2026-04-16, OPEN, +1873/-2, 12 files)
- Title: "Restore Double Sparsity attention backend on latest main"
- Verbatim from the PR body: reintroduces legacy `double_sparsity_backend.py` and `DoubleSparseTokenToKVPool`; fixes piecewise CUDA-graph auto-disable, forward-kwargs drift, KV pool memory accounting, output view dim bug.
- Tested on Llama only via `test/manual/test_double_sparsity.py` (MMLU 0.75 > 0.65 threshold).
- Performance: 3-12% throughput REGRESSION vs Triton baseline on H100 (no CUDA graph, torch.gather/topk overhead). PR body explicitly says "Performance optimization is planned for follow-up work."
- Page-size: legacy backend is page=1 only (predates HiSparse paging). No MLA support. No FP8.

PR #25304 — `dev/double-sparsity-v2` (created 2026-05-14, OPEN, +22552/-8, 90 files, no description, no CI)
- 50+ commits, organized as M1..M9 milestones plus v1.1 fix train and v2 native-decode pivot.
- M1: skeleton + calibration schema + TP. M2: K_label storage + write kernels. M3: selection pipeline (torch ref + Triton). M4: DS FA3 adaptor + CUDA-graph correctness. M5: generic-coordinator wiring. M6: smoke. M7: calibration script. M8/M9: bench + ship-gate.
- v1.1: vectorize selection, lift CUDA-graph restriction, save_kv_cache real-path threading, effective_sparse_mask hook, BLOCK_T/K_block knobs, stage-1 + stage-2 block-topk Triton kernels, score-aware union + CUDA-graph capture/replay.
- **v2 pivot**: "native sparse-decode kernels replace FA3 page-table path" (commit `30ba60dae`). The v2 native path is "capture-safe dispatch, validated end-to-end at 16K" (commit `91960f63a`).
- Final claims: "BOTH GATES PASS at conc=32/128K/tb=8192" (commit `faa8a5d6a`), Pareto curve confirmed; NIAH retrieval validated.
- Backbone: FA3 (`FlashAttention 3`) + native sparse-decode kernel. **No MLA / DeepSeek-V3.2 path.** Selection driven by user's own coordinator wiring (not the HiSparse `SparseCoordinator`); FlashInfer top_k_page_table is a pluggable selector backend in the final commits.
- The PR's 90 files include `HANDOFF_NATIVE.md`, `SESSION_REPORT_2026-05-14.md`, pensieve installs, NIAH calibration scripts, bench harnesses — much of this is workspace/session noise rather than shippable upstream code.

**Benchmark (development/benchmark.sh):** uses `--dataset-name generated-shared-prefix` with SYS_LEN=2253, Q_LEN=1843 (ISL=4096), OSL=512, GSP_NUM_GROUPS=1, NUM_PROMPTS=5*64=320, concurrencies 16/32/64 (only 64 active). Targets a single sglang server on PORT=30000.

## 3) Specific Open Questions to Tackle

Critique the draft and answer:

a. **Does Double Sparsity even make semantic sense on top of DeepSeek-V3.2-Exp (which is already DSA / NSA-sparse)?** Is it:
   - i) Redundant — DSA already token-selects, DS would just be a second filter on a small set
   - ii) Complementary — DS provides channel-level sparsity for the dense paths inside NSA's indexer/compressor
   - iii) A replacement — substitute a DS-style label cache for the NSA indexer's learned head
   - iv) Something else
   Which interpretation matches the client's likely intent given they asked for "Double Sparsity on DeepSeek-V3.2"?

b. **Resume vs restart, given the verified state of PR #22992 and PR #25304.** The user's lean toward restart is plausible — but what is the cost of throwing away the M3/v1.1/v2-native selection kernels in PR #25304? Should the plan say "restart but cherry-pick the kernels"?

c. **What is the right *upstream-shaped* design?** Options:
   - Plug Double Sparsity into HiSparse as a third algorithm (`double_sparsity`) alongside `quest` and `deepseek_nsa`
   - Build a parallel framework
   - Build a DeepSeek-V3.2-specific path that reuses `NSABackendAdaptor` (currently a TODO stub) by completing it
   - Direct attention-backend integration (legacy style, bypass HiSparse)

d. **SLO realism.** 30 tok/s and P99 TTFT < 22s at conc=64, ISL=4096, OSL=512, 55% prefix cache hit. On what hardware (the draft doesn't say)? Is this per-request output throughput or aggregate? At conc=64, per-request 30 tok/s implies aggregate ~1920 tok/s decode and 22s TTFT means up to 22s allowed for prompt processing of 4096 tokens. Is this aggressive, baseline, or trivial for DeepSeek-V3.2-FP8? What would sparse vs dense each project? Does Double Sparsity actually help at this workload (long ISL but very short OSL, lots of prefix cache hit), or is the bottleneck prefill not decode?

e. **Page size 64**, "technically not explicitly listed as a hard requirement, but significantly preferred." HiSparse on V3.2 already uses page=64 (`flashmla_kv` requires page=64). Is "support different page sizes" cheap to add or load-bearing? What's the test matrix?

f. **NDA / labels**. Double Sparsity needs an **offline-calibrated label cache** (the channel sparsity). Where does this calibration happen for DeepSeek-V3.2? Who runs it, against what dataset, and how is the artifact distributed? The draft is silent on calibration ownership.

g. **Deferred reqs (GLM-5, 128k ISL, nvfp4/mxfp4) and downstream reqs (Twilight top-p, Extensions, PD-Disagg+HiSparse integration).** Which of these constrain the *initial* design? E.g. if Twilight (top-p) is on the roadmap, the selection kernel should be top-p-shaped from day one, not top-k baked in.

h. **The "Extensions" downstream requirement** is vague. What does "Extensions as a general knob for the sglang engine" mean concretely? Is this asking for a plugin system separate from HiSparse?

i. **Risks not in the draft**: CUDA-graph capture for variable-K selection, FP8 KV requantization on selected pages, label cache memory cost at TP, schedule_batch metadata threading, PD-disagg label-cache transfer, observability (per-request top-k hit rate), quality regressions (MMLU/NIAH gates).

## 4) Output Format

Return your analysis in EXACTLY these sections (use these literal headers):

```
CORE_RISKS:
- ...

MISSING_REQUIREMENTS:
- ...

TECHNICAL_GAPS:
- ...

ALTERNATIVE_DIRECTIONS:
- ...

QUESTIONS_FOR_USER:
- ...

CANDIDATE_CRITERIA:
- AC-1: ...
- AC-2: ...
- ...
```

For ALTERNATIVE_DIRECTIONS, give 2-3 concrete options, each with a one-line tradeoff summary. For QUESTIONS_FOR_USER, list questions that require a human decision (e.g. "Is 30 tok/s aggregate or per-request?"). For CANDIDATE_CRITERIA, propose AC items that the plan should adopt; favor testable criteria with positive/negative tests.

Be concrete. Cite filenames or commit SHAs where helpful. Prefer "here is the gap" over "consider whether there might be a gap." Push back where the draft is hand-wavy. The user explicitly invited dissent ("Help me first decide…").

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-19_09-02-49
- Tool: codex
