CORE_RISKS:
- The central ambiguity is semantic: Double Sparsity should not be blindly stacked after DeepSeek-V3.2 NSA/DSA. That would likely be redundant and may quality-regress by applying a second token filter to an already sparse set.
- The likely client intent is “Double Sparsity as an alternative selector/label-cache path for DeepSeek-V3.2 sparse attention,” not the old Llama dense-attention backend. Confirm this before coding.
- Restarting clean is correct, but throwing away PR #25304 entirely is wasteful. The plan should say: restart on current main, then audit/cherry-pick the useful selection kernels from #25304, especially M3/v1.1/v2-native work and commits `30ba60dae`, `91960f63a`, `faa8a5d6a`.
- PR #22992 is not a viable base: page=1, no MLA, no FP8, Llama-only, and documented 3-12% throughput regression.
- PR #25304 is not a viable upstream base as-is: huge diff, no PR description/CI, workspace noise, custom coordinator, FA3/native-decode focus, and no DeepSeek-V3.2 MLA path.
- The current HiSparse validator conflicts with the workload: it requires `--disable-radix-cache`, while the target benchmark assumes ~55% prefix cache hit.
- SLOs are underspecified. “30 tokens/s at concurrency 64” is meaningless without hardware, TP/EP layout, per-request vs aggregate, and whether TTFT includes queueing.
- Double Sparsity may not help the stated workload if the bottleneck is prefill/cache behavior rather than decode attention. A dense/native-NSA baseline is mandatory.

MISSING_REQUIREMENTS:
- Hardware target: GPU type/count, HBM size, interconnect, TP/EP/DP config, CUDA version, and expected deployment topology.
- Throughput definition: per-request output tok/s, aggregate decode tok/s, or benchmark-reported normalized tok/s.
- Exact model revision: `deepseek-ai/DeepSeek-V3.2` is not enough; FP8 checkpoint revision and SGLang model path matter.
- Whether “Double Sparsity on V3.2” means replacing NSA selection, augmenting NSA internals, or exposing DS as a separate selectable sparse algorithm.
- Calibration ownership: who runs offline label calibration, on what dataset, with what quality target, and how the artifact is versioned/distributed.
- Label artifact format: safetensors/json/binary, model-revision checksum, TP compatibility, page-size compatibility, dtype compatibility, and failure behavior when missing.
- Cache requirement: whether radix/prefix cache must remain enabled for the client benchmark.
- Page-size contract: whether only page=64 is required for V3.2, or whether arbitrary page sizes must be performant.
- Quality gates: MMLU alone is insufficient. Need NIAH/needle, long-context retrieval, perplexity or task accuracy, and regression thresholds against native NSA.
- Observability requirements: selected pages/tokens, effective sparsity, top-k/top-p hit rate, fallback count, graph-capture status, and per-request quality/debug traces.
- PD-disagg requirements: whether labels move with KV, are recomputed on decode workers, or are replicated out-of-band.

TECHNICAL_GAPS:
- `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py`: `NSABackendAdaptor.adapt_for_attn_metadata` is a TODO, so HiSparse-style selectors cannot currently drive the MLA sparse path.
- `SparseConfig` already has `algorithm`, `backend`, `page_size`, `top_k`, and `sparse_extra_config`; building a parallel DS framework would duplicate existing control surfaces.
- DeepSeek-V3.2 already has native sparse machinery under `python/sglang/srt/layers/attention/nsa/` and `dsv4/`; DS must interoperate with or replace that metadata path.
- The plan does not address FP8 KV handling: selected pages may require dequant/requant, layout-aware gathers, and correctness tests for `fp8_e4m3`.
- CUDA graph capture is a first-class risk for variable-K/top-p selection. The API should be max-K/static-buffer shaped from day one, even if initial policy is top-k.
- Page size 64 is not just a preference for V3.2: `flashmla_kv` currently expects it. “Support different page sizes” is likely a separate kernel/metadata test matrix, not a cheap config toggle.
- Schedule-batch metadata threading is under-specified: DS needs selector output to reach attention metadata without breaking batching, capture, or mixed sparse/dense requests.
- The draft does not define negative behavior: unsupported model, missing labels, wrong page size, wrong dtype, radix-cache conflict, or graph capture failure.
- PR #25304’s kernels may be valuable, but the FA3/native-decode assumptions need explicit adaptation to MLA/FlashMLA before they count toward the client path.
- Twilight/top-p is deferred, but it constrains the selector ABI now. Do not bake “fixed top-k only” into metadata structures.

ALTERNATIVE_DIRECTIONS:
- Clean restart + HiSparse integration + cherry-picked kernels: best upstream shape; reuse #25304 kernel work while implementing `double_sparsity` as a HiSparse algorithm and completing the MLA adaptor.
- DeepSeek-V3.2-specific client path behind HiSparse config: fastest route to the immediate SLO; higher risk that GLM-5, Twilight, and PD-disagg require later rework.
- Resume #25304 and carve it down: preserves the most existing implementation, but review risk is high because it lacks MLA support, uses custom coordination, and contains substantial non-shippable noise.

QUESTIONS_FOR_USER:
- Is 30 tok/s per request or aggregate across concurrency 64?
- What exact hardware and parallelism must satisfy the SLO?
- Does P99 TTFT include queueing, prefix-cache lookup, and scheduler wait time?
- Must radix/prefix cache remain enabled for the benchmark?
- Should DS replace DeepSeek-V3.2 NSA selection, augment it, or be independently selectable for A/B testing?
- Who owns offline label calibration and artifact distribution?
- Is page=64 sufficient for the first deliverable on DeepSeek-V3.2?
- What is the minimum acceptable quality delta versus native NSA?
- Are GLM-5 and 128k ISL committed follow-ons or only possible roadmap items?
- What does “Extensions as a general knob” mean concretely: HiSparse config extension, engine plugin system, or per-request runtime knob?

CANDIDATE_CRITERIA:
- AC-1: `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity",...}'` works on DeepSeek-V3.2 FP8 through the HiSparse path, not a separate DS-only CLI.
- AC-2: Unsupported models/dtypes/page sizes fail with explicit errors; no silent dense fallback.
- AC-3: Missing or mismatched label-cache artifact fails before serving starts, with model revision, dtype, TP size, and page-size validation.
- AC-4: Page size 64 passes on DeepSeek-V3.2 FP8; at least one negative test proves invalid page sizes are rejected or routed to a documented fallback.
- AC-5: Radix-cache behavior is explicitly tested: either DS works with the target 55% prefix-cache hit workload, or the SLO benchmark is redefined without cache hits.
- AC-6: CUDA graph capture/replay passes for concurrency 16/32/64 with stable buffers and no data-dependent allocation in the decode path.
- AC-7: Benchmark reports dense/native-NSA baseline and DS result on the same hardware, same cache setting, same model revision, same workload.
- AC-8: Meets or beats 30 tok/s and P99 TTFT < 22s under the clarified throughput definition at concurrency 64.
- AC-9: Quality gate passes against native NSA with agreed thresholds on NIAH/retrieval plus at least one general benchmark such as MMLU or perplexity.
- AC-10: Observability exposes selected token/page count, effective sparsity, label-cache hit/miss or validity, graph-capture status, and fallback count.
- AC-11: Selection ABI supports both fixed top-k and future bounded top-p/Twilight without changing attention metadata layout.
- AC-12: PR is upstream-reviewable: minimal branch, no session artifacts, focused tests, and #25304 code imported only where it directly supports the MLA/HiSparse path.
