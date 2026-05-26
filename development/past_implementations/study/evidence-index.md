# Phase-1 Evidence Index

This index is the traceability backbone for the Phase-2 design documents.
It does **not** restate findings; it cites them. For each of the three
implementations there is one table whose rows enumerate the same set of
topics, so the reader can read horizontally across all three repos.

Conventions:

- File paths are **repo-relative** to each implementation's root, which lives
  under `development/past_implementations/<Impl>/`.
- `Observed:` = statement directly grounded in code/config/test artifact.
- `Inferred:` = strongly suggested by code shape or convention but not
  spelled out by the author.
- Line numbers refer to the snapshots checked into
  `development/past_implementations/` (read-only inputs).

The three implementations under study are:

- **Implementation A — DoubleSparse**
  (`development/past_implementations/DoubleSparse/`): the original research
  repository accompanying paper 2408.07092. Standalone GPT-Fast-style
  inference loop plus an HF-adapter eval harness and a CPU-offload variant.
- **Implementation B — sglang-last-with-double-sparsity**
  (`development/past_implementations/sglang-last-with-double-sparsity/`):
  a snapshot of upstream SGLang taken just **before** commit `44e67c683`
  (#23009, 2026-04-17) which removed the double-sparsity feature. This is the
  most directly relevant codebase for the porting target.
- **Implementation C — Twilight**
  (`development/past_implementations/Twilight/`): a generalised
  sparse-attention research framework that includes double-sparsity as one
  of several pluggable selectors, with a novel "weight pruner" stage added
  after the selector.

---

## Implementation A — DoubleSparse

Standalone PyTorch + Triton repo, not a fork of any serving framework.
Channel calibration lives here; this is the canonical artifact source the
other two repos depend on.

| Topic / Claim Area | Key Files & Symbols (with line numbers where possible) | What Can Be Directly Concluded | What Remains Uncertain | Suggested Next Inspection Step |
|---|---|---|---|---|
| Framework lineage / repo type | Repo root (`README.md`, `requirements.txt`); `models/` directory (GPT-Fast style); `evaluation/` (HF adapters) | Observed: standalone repo, loosely modelled on Meta GPT-Fast; not a fork of HF/vLLM/SGLang. Pins `torch==2.5.1`, `triton==3.1.0`, `transformers==4.47.1`, CUDA 12.1. | Was the GPT-Fast lineage copied wholesale or partially rewritten? | Diff `models/model.py` against upstream `pytorch-labs/gpt-fast` to quantify divergence. |
| Entry points (calibration, inference, eval, benchmark) | `config/offline_calibration.py`; `models/generate.py`; `evaluation/perplexity_eval.py`, `mmlu.py`, `retrieval_eval.py`, `chat.py`; `AIME/aime.py`; `LongBench/pred.py`; `scripts/run_attn.sh` -> `models/triton_kernels/attention.py` | Observed: four distinct entry families — calibration, standalone generate, HF-adapter eval, and operator microbenchmark. Each has its own `argparse`. | Is the standalone generate path actively maintained or only an artifact? It has no smoke test. | Run `models/generate.py` end-to-end on a small Llama-2 checkpoint to confirm it still loads weights and produces tokens. |
| Channel-config artifact format & generation | `config/offline_calibration.py:get_qk_hook` (lines 91-93); 13+ committed JSONs under `config/<owner>/<name>.json` (e.g. `config/meta-llama/Llama-2-7b-hf.json` ~3.1 MB) | Observed: per-layer-per-head sorted channel indices of length `head_dim`. Metric: `mean_t(|Q_t * K_t|)` over 256 Pile-val samples, 512-token block. Key format `model.layers.{i}.self_attn.{q\|k\|qk}_proj`. | Methods 3-5 (commented out in `get_qk_hook`) — why discarded? Are k-only configs better than qk-merged in practice? | Re-run calibration with one of the alternative scoring metrics and compare top-channel overlap. |
| Label cache representation | `models/model.py:71-96` `KVCache`; populated at `models/model.py:245-246` via `get_label_tensor(...)`; offloading variant `offloading/model.py:82-148` | Observed: label buffer shape `[B, S, H, heavy_channel_num]`, fp16 by default; populated every token during prefill. Offloading variant keeps label on GPU, full K/V on pinned CPU. | Whether `--q_bits` int8/int4 label quantisation path was ever validated (the kernel exists in `bgemv_int8.py`). | Force `--q_bits 8` on an eval and compare PPL delta. |
| Token-level top-k selection mechanism | Standalone: `models/model.py:265-294 sparse_forward()`; HF adapter: `evaluation/modify_llama.py:83-273 LlamaAttention_heavy_hitter.forward`; Offloading: `offloading/model.py:82-148` | Observed: three distinct selection implementations coexist. Standalone uses `torch.topk(label_scores, heavy_const, dim=-1)` then a sparse Triton kernel. HF adapter computes scores and masks non-top-k with `-inf`, keeping dense softmax. Offloading does topk then async fetch. | Numerical equivalence between the three paths is not asserted anywhere. | Add a unit test that runs all three paths on the same prompt and compares output logits. |
| Custom kernels (language, shapes/dtypes) | `models/triton_kernels/sparse.py:107-152 fwd_sparse_no_mask_kernel`; `models/triton_kernels/channel.py:10-30 get_label_tensor_kernel`; `bgemv.py`, `bgemv_int8.py`, `argsort.py`, `quantize.py`, `sparq.py`, `heavy.py`, `attention.py` | Observed: Triton only, no CUDA C++. `fwd_sparse_no_mask` gathers K/V at heavy-list indices per (head, batch). Auxiliary kernels for label-cache extraction and int8 BGEMV are present. | Several kernels (sparq, heavy, argsort) appear to be legacy/alternative experiments; which are reachable from the main path? | Static-trace each entry point and mark kernels as live vs dead. |
| KV cache layout / memory pool | Standalone: `models/model.py:71-96 KVCache.__init__` (`k_cache`, `v_cache`, `k_label` as registered buffers). Offloading: `offloading/model.py:82-148` pinned-CPU `k_full,v_full` + GPU `k_compressed,v_compressed,k_label` | Observed: per-layer Python module owns its own K/V/label tensors. No global pool, no paging. Offloading variant maintains parallel CPU/GPU buffers and uses DGL `gather_pinned_tensor_rows`. | No streaming/prefill chunking — what is the practical max context? | Trace memory growth on a 128 k-context Llama and confirm OOM threshold. |
| Attention backend / dispatch routing | `models/model.py:Attention.forward` (selects between `dense_forward` and `sparse_forward` based on accumulated tokens); HF adapter monkey-patches `LlamaAttention.forward` via `convert_kvcache_llama_heavy_recent` (`evaluation/modify_llama.py:277+`) | Observed: routing is per-layer module-level. Standalone: prefill stays dense until `heavy_const` tokens accumulate, then switches to sparse decode. HF adapter: layers `>=2` use heavy-hitter forward; layers 0-1 dense. | Is the `>=2` threshold magic-numbered or paper-derived? | Search referenced paper section 4 for "first two layers"; cross-check vs Twilight's `skip_first_two_layers` flag. |
| Prefill vs decode treatment | `models/model.py:sparse_forward` is decode-path only; prefill runs full dense attention writing into `k_cache,v_cache,k_label` | Observed: prefill always dense; decode is the only sparse stage. | No chunked-prefill experiments; behaviour under long prompts not measured. | Add a long-prompt eval (~32 k tokens) and instrument prefill latency. |
| Models supported & their wiring | `evaluation/modify_llama.py`, `modify_mistral.py`, `modify_mixtral.py`, `modify_qwen2.py`; standalone `models/model.py` (Llama only) | Observed: LLaMA 7B/13B/70B, Llama-3.1-8B, Llama-3.2-11B-Vision, Mistral 7B, Mixtral 8x7B, Qwen2 7B/32B, DeepSeek-R1-Distill-Qwen-32B, Vicuna 7B, LongChat 7B. Wiring is per-model HF-adapter monkey-patch via `convert_*` helpers. | Mixtral/Vision support depth — does it cover MoE routing or vision encoder paths? | Open `modify_mixtral.py` and confirm MoE expert dispatch is preserved. |
| Tests / accuracy validation | `evaluation/perplexity_eval.py` (WikiText-2); `evaluation/mmlu.py`; `evaluation/retrieval_eval.py` (LONGEVAL); `LongBench/pred.py` (13 tasks); `AIME/aime.py`; `evaluation/chat.py` | Observed: rich accuracy eval surface but no `pytest`-style unit tests. Validation is by full-pipeline accuracy runs. | No assertion of numeric equivalence between sparse and dense paths. | Add a tiny `test_equivalence.py` that runs a 128-token decode dense vs sparse with `heavy_const = seq_len` and asserts equality. |
| Benchmarks (operator-level and end-to-end) | `scripts/run_attn.sh` -> `models/triton_kernels/attention.py:att()`; sweeps batch in {1,4,8,16,32}, ctx in {2048,4096,8192,16384}; 1000 warmup + 1000 measured | Observed: operator-level attention latency only. No end-to-end throughput/latency benchmark script. | Are paper headline numbers reproducible from these scripts alone? | Run `run_attn.sh` and compare to paper table 3. |
| Reproducibility (pinned deps, seeds, datasets, configs) | `requirements.txt`; `config/*.json` (committed); calibration dataset = Pile-val hard-coded | Observed: deps pinned (`torch==2.5.1`, `triton==3.1.0`, `transformers==4.47.1`, `datasets==3.2.0`, xformers 0.0.28.post3, DGL cu121). 13+ pre-computed channel configs committed. | Calibration seed not pinned in `offline_calibration.py`; random sample selection. | Add `--seed` argument and re-run for two seeds to estimate config stability. |
| Hardware assumptions (CUDA, SM, etc.) | `requirements.txt` (`nvidia-cuda-runtime-cu12==12.4.127`); `requirements.txt` DGL `dgl-cu121` | Observed: CUDA 12.1 toolchain, NVIDIA-only. No HIP/CPU fallback. SM not pinned. | Tested GPU model unrecorded (likely A100/H100). | Grep for `compute_capability`/`sm_` hardcoded values in Triton kernels. |
| CPU offloading / multi-tier memory | `offloading/model.py:82-148 KVCache`; `evaluation/offload_llama.py:convert_kvcache_llama_offloading()`; entry via `perplexity_eval.py --mode ds-offload` | Observed: full K/V live pinned on CPU; compressed `[B,H,heavy_const,D]` GPU buffers act as working set; DGL `gather_pinned_tensor_rows` fetches selected tokens. | Whether the gather is truly async (CUDA stream overlap) or blocks the decode kernel. | Insert NVTX markers and read a Nsight Systems timeline. |
| Multi-GPU / TP / PP support | Repo-wide grep: no `torch.distributed`, no `device_map`, no TP wrappers (only HF `device_map="auto"` indirectly via transformers) | Observed: single-GPU only. No tensor or pipeline parallelism in the standalone or adapter paths. | Mixtral might rely on HF transformers' built-in MoE; not verified. | Grep for `device_map` / `accelerate` usage in `evaluation/*.py`. |
| CUDA-graph compatibility | None. Standalone uses `torch.compile` selectively (`models/model.py`); HF adapter does not. | Observed: no `torch.cuda.CUDAGraph` use anywhere. | Could `torch.compile` over the sparse decode path generate stable graphs? | Try `torch.compile(mode='reduce-overhead')` on `sparse_forward` and inspect. |
| Quantization integration | `--q_bits` flag in `generate.py`; `bgemv_int8.py`, `quantize.py` Triton kernels | Observed: label-cache int8/int4 path implemented at kernel level but no end-to-end eval committed (no JSON config under `config/quant_*`). | Effect on accuracy unmeasured. | Run `perplexity_eval.py` with `--q_bits 8` and `4` and report PPL delta. |
| Relation to the reference paper | Paper 2408.07092 (Yang et al., 2024) authored by same group (`config/` paths reference `andy-yang-1`) | Observed: this is the **canonical reference implementation**. Paper's Algorithm 1 (offline channel selection by mean \|Q*K\|) maps to `get_qk_hook` line 91-93. Algorithm 2 (token top-k via label cache) maps to `sparse_forward` line 265-294. | Paper section 4.3 ("dynamic heavy-token budget") not implemented in code — `heavy_const` is static. | Read paper section 4.3 and confirm. |
| Notable absences / unfinished pieces | Search for `TODO`/`FIXME` markers; `sparq.py`/`heavy.py` in `triton_kernels/` | Observed: no training/fine-tuning code; only inference. Several Triton kernels (sparq, heavy) appear unreached. No dynamic `heavy_const`. No CUDA-graph wrapping. | Whether `sparq.py` was ever benchmarked head-to-head. | Run `bench_gemv` analogue manually. |

---

## Implementation B — sglang-last-with-double-sparsity

The most directly relevant codebase for the porting effort: a real SGLang
snapshot from immediately before the feature was removed. **This table is
intentionally the most detailed of the three.**

| Topic / Claim Area | Key Files & Symbols (with line numbers where possible) | What Can Be Directly Concluded | What Remains Uncertain | Suggested Next Inspection Step |
|---|---|---|---|---|
| Framework lineage / repo type | `python/sglang/version.py:1-24` (setuptools_scm); root `pyproject.toml`; `.git/` history with `44e67c683` (#23009, 2026-04-17 "Remove deprecated double sparsity feature") | Observed: this is upstream SGLang at the commit immediately before DS removal; not a fork. Total DS-added LOC across all commits is ~1,700. Relevant prior commits: `061e54631` (initial #1459), `ea34350d8` (#2188 rename), `ad26f298e` (#6905 init fix). | Why was the feature removed — unstated in commit msg. Maintenance burden, low usage, or superseded by something else? | Search issue tracker for #23009 discussion, ping maintainers. |
| Entry points (calibration, inference, eval, benchmark) | `test/manual/test_double_sparsity.py` (launches the SGLang server with DS flags and runs MMLU); no calibration code (delegated to Implementation A); benchmarks via standard SGLang `bench_serving`/`bench_one_batch` paths | Observed: DS in SGLang is **inference-only**. Calibration is delegated to external DoubleSparse repo (URL cited in `test/manual/test_double_sparsity.py:25`). The only DS-specific test is `test_double_sparsity.py`. | Whether DS interacts cleanly with `bench_serving` (no DS-specific benchmark in tree). | Run `python -m sglang.bench_one_batch --enable-double-sparsity ...` and capture metrics. |
| Channel-config artifact format & generation | `python/sglang/srt/model_executor/model_runner.py:2160-2171 init_double_sparsity_channel_config()`; example `test/manual/double-sparsity-config-Llama-3.1-8B-Instruct.json` (~900 KB) | Observed: identical JSON format to Implementation A — keys `model.layers.<i>.self_attn.<q\|k>_proj`, values = per-head channel-index lists of length `head_dim`. Only first `ds_heavy_channel_num` indices retained (line 2167). Loaded into `self.sorted_channels` as per-layer CUDA tensors of shape `[num_heads, heavy_channel_num]`. | No version/format check on the JSON — older/newer configs would silently load. | Add a magic-version field assertion. |
| Label cache representation | `python/sglang/srt/mem_cache/memory_pool.py:1972-2060 DoubleSparseTokenToKVPool`; instantiated in `model_runner_kv_cache_mixin.py:412-425` | Observed: extends the SGLang token-to-KV pool with a third buffer per layer. Shapes: `k_buffer[size+page_size, head_num, head_dim]`, `v_buffer[size+page_size, head_num, head_dim]`, `label_buffer[size+1, head_num, heavy_channel_num]`. New API surface: `get_label_buffer(layer_id)` (line 2038), `set_kv_buffer(layer, loc, cache_k, cache_v, cache_label)` (line 2047-2059). | Why label buffer is `size+1` while K/V are `size+page_size` — possible off-by-one or intentional sentinel slot. | Inspect call sites of `get_label_buffer` and confirm. |
| Token-level top-k selection mechanism | `python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py` stages 1-3: `_sparse_fwd_kernel_flash_decode_stage1` (329-398), CPU `torch.topk` at line 759, `_sparse_fwd_kernel_flash_decode_stage2` (400-513), `_sparse_fwd_kernel_flash_decode_stage3` (517-558); orchestrator `flash_decode_sparse_attention_fwd` (700-776) | Observed: a **3-stage Triton pipeline**. Stage 1 BGEMV `Q_label @ K_label^T` -> approx-logits `[head, batch, seq_len]`. Stage 2 selects top-k tokens via host-side `torch.topk(att_out_approx, heavy_token_num, dim=-1).indices` -> `[num_head, batch, heavy_token_num]`. Stage 3 does full Q,K,V attention restricted to selected positions (gathered via `req_to_token` + topk indices). A final block-reduction kernel (Welford) produces `O`. | The host-side `torch.topk` round-trip is a known bottleneck — but unmeasured. Stage 1 output dtype/precision implications also unmeasured. | NVTX-instrument the three stages and the topk; measure proportion of decode latency. |
| Custom kernels (language, shapes/dtypes) | `python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py` (~1,106 lines); dense fallback `_fwd_kernel_flash_decode_stage1/2` (29-188), wrappers `flash_decode_attention_fwd` (284-326); extend kernel `_fwd_kernel` (782-990), wrapper `extend_attention_fwd` (996-1105) | Observed: Triton only. Head dims supported by stages 1-3 = {16, 32, 64, 128}; extend kernel additionally supports {192, 256, 288, 576}. SM-tuned `BLOCK_M/BLOCK_N` (lines 1040-1053) with separate paths for Hopper SM9.0, Ampere SM8.x, and HIP (AMD). Extend kernel adapted from SGLang v0.4.2 reference. | DeepSeek-V3.2 / MLA head dim 576 lands in the extend kernel's branch but no sparse-decode branch for it. | Trace MLA forward path and confirm whether `decode_sparse_attention_fwd` would even be called. |
| KV cache layout / memory pool | `python/sglang/srt/mem_cache/memory_pool.py:1972-2060`; wired in `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:21,412-425` | Observed: `DoubleSparseTokenToKVPool` is selected via `model_runner_kv_cache_mixin._create_token_to_kv_pool` when `--enable-double-sparsity` is set. The pool layers on top of the standard token pool by adding a parallel `label_buffer` array; allocation/free routines reuse parent. | Interaction with RadixAttention prefix cache — does the prefix-share path also share label tokens? | Trace `RadixCache.match_prefix` and confirm label-buffer indexing matches K/V indexing. |
| Attention backend / dispatch routing | `python/sglang/srt/layers/attention/double_sparsity_backend.py` (entire 258 lines); registered in `python/sglang/srt/layers/attention/attention_registry.py:101-106`; enforced backend = `"triton"` in `model_runner.py:927-932` | Observed: dedicated `DoubleSparseAttnBackend` class. `forward_extend()` (113-165) is always dense and writes K/V/K_label to pool. `forward_decode()` (167-257) routes per-batch: `if (min_seq_len < heavy_token_num or max_seq_len < sparse_decode_threshold)` -> dense fallback (`decode_attention_fwd`), else sparse 3-stage. Backend choice is forced when `--enable-double-sparsity` is on. | The routing is **per-batch**, not per-request — a heterogeneous batch where some requests are short and some long is silently routed to dense. | Add a per-request routing branch and benchmark mixed-length workloads. |
| Prefill vs decode treatment | `double_sparsity_backend.py:forward_extend()` 113-165 — always dense and writes label channels alongside K/V via lines 128-134 then 137-139 (`set_kv_buffer(..., cache_label)`); decode at `forward_decode()` 167-257 | Observed: prefill is **always dense**. Label channels are extracted from K during prefill (lines 128-134) so that decode has the label cache pre-populated. Sparsity only fires on decode. | Why prefill stays dense — no comment; consistent with paper but not stated. | Confirm by reading paper section 4. |
| Models supported & their wiring | No per-model changes; CLI flag works transparently on any model with standard `model.layers.<i>.self_attn.{q,k}_proj` naming and a Triton-compatible attention path. Channel-config consumer: `model_runner.py:668-674, 2156-2171` | Observed: zero model-side edits. Anything with LLaMA-style projection naming and dense MHA/GQA Triton attention works. DeepSeek-V3.2 (MLA) and GLM-5.1 (GLU/RoPE variants) are **not** wired — MLA naming differs and the kernel head dims don't map cleanly to the sparse-decode kernel. | Whether MoE models (Mixtral, Qwen2-MoE) silently work — they route through the same attention but per-expert weights might confuse the channel-config keys. | Try launching with a Mixtral channel config and confirm. |
| Tests / accuracy validation | `test/manual/test_double_sparsity.py` (66 lines); ref config `test/manual/double-sparsity-config-Llama-3.1-8B-Instruct.json` | Observed: a single MMLU-subset smoke test (64 examples, 32 threads, accept >= 0.65) launching the server with `--enable-double-sparsity --ds-channel-config-path ... --ds-heavy-channel-num 32 --ds-heavy-channel-type k --ds-heavy-token-num 512 --ds-sparse-decode-threshold 0 --max-total-tokens 200000`. No CI; lives under `test/manual/`. | No numerical-equivalence test (sparse vs dense on the same prompt). | Add a `test_ds_equivalence.py` that compares logits at `heavy_token_num >= max_seq_len`. |
| Benchmarks (operator-level and end-to-end) | None DS-specific. SGLang has generic `python/sglang/bench_one_batch.py`, `python/sglang/bench_serving.py` but no DS-tagged recipe. | Observed: no DS-specific benchmark script; no recorded baselines committed. | No published throughput/latency numbers for DS-on-SGLang. | Run `bench_one_batch` with and without `--enable-double-sparsity` on a Llama-3.1-8B and report decode latency. |
| Reproducibility (pinned deps, seeds, datasets, configs) | `pyproject.toml` (SGLang deps); `test/manual/double-sparsity-config-Llama-3.1-8B-Instruct.json`; no DS-specific seed | Observed: deps governed by SGLang's main `pyproject.toml` — no DS pin-set. One example channel config committed. Calibration dataset/seed live in Implementation A (external). | DS config drift if regenerated with newer Implementation A. | Pin a checksum of the example config in the test. |
| Hardware assumptions (CUDA, SM, etc.) | `double_sparsity_attention.py:1040-1053` (SM dispatch); `requirements*.txt` of SGLang | Observed: CUDA via Triton (no CUDA toolkit pin in DS code itself); SM 8.x and 9.0 tuned; HIP fallback present. No `flash-attn` dependency in the DS path. | Performance on SM 7.x untested. | Run on a V100 (SM 7.0) and observe. |
| CPU offloading / multi-tier memory | Repo-wide grep in `python/sglang/srt/mem_cache/`: no DS-specific offload code | Observed: **no CPU offloading**. Label cache lives on GPU; full K/V remains in pool. Implementation A's offload path was not ported. | Whether SGLang's HiCache could host the label cache transparently. | Inspect `python/sglang/srt/mem_cache/hicache*.py` for layer-key hooks. |
| Multi-GPU / TP / PP support | No DS-specific TP/PP code; backend just consumes `model_runner.tp_size` indirectly via attention layer | Observed: **not validated**. The backend should be transparent to TP because per-layer config is per-head and partitioning is per-head, but no test exercises TP. | Whether channel-config head ordering matches the SGLang TP head partition. | Launch with `--tp 2` and verify per-rank head indices line up. |
| CUDA-graph compatibility | `python/sglang/srt/model_executor/model_runner.py:927-932` — `disable_cuda_graph = True` is forced when `--enable-double-sparsity` is set | Observed: **CUDA graphs are explicitly disabled**. Likely because the host-side `torch.topk` at stage 2 produces variable-shape index tensors per step. | Whether a fixed-shape padded variant of stage 2 could re-enable graphs. | Try replacing host `torch.topk` with a fused Triton topk and re-enable graphs. |
| Quantization integration | None DS-side. SGLang has fp8/awq paths elsewhere but `DoubleSparseTokenToKVPool` only handles fp16/bf16 buffers (parent class default) | Observed: no fp8 KV or int8 label support in the DS pool. Implementation A's int8 BGEMV was not ported. | How DS would compose with SGLang fp8 KV cache. | Attempt to set `--kv-cache-dtype fp8_e5m2` together with `--enable-double-sparsity` and observe failure mode. |
| Relation to the reference paper | External calibration via Implementation A; runtime sparse decode = paper Algorithm 2; prefill-always-dense matches paper convention | Observed: this is a **production-grade port** of the paper, not a research artifact. Coverage matches paper Algorithm 1+2; dynamic `heavy_token_num` (paper section 4.3) is **not** implemented (`heavy_token_num` is a static CLI flag). | Paper headline accuracy/latency reproducible from this code? Not measured here. | Compare against paper table 5 on Llama-2-7B WikiText-2. |
| Notable absences / unfinished pieces | Removed in commit `44e67c683` (#23009); only `test/manual/test_double_sparsity.py`; no CI; no MLA support; no fp8/int8 KV; no CUDA graphs; no offload; comment in code hints at future dynamic `heavy_token_num` but unimplemented | Observed: significant gaps that block landing on modern models — (1) no MLA/non-LLaMA projection support, (2) CUDA graphs disabled, (3) host-side topk forces a sync, (4) no quantisation integration, (5) per-batch routing only, (6) no CI test. | The removal in #23009 strongly suggests these gaps drove the deprecation — but the commit message is silent. | Read full PR discussion on #23009. |

---

## Implementation C — Twilight

A generalised sparse-attention research framework (the paper-2408.07092
author's *follow-on* line of work) where double-sparsity is one of several
selector strategies, and the novel contribution is the post-selector
**weight pruner** stage.

| Topic / Claim Area | Key Files & Symbols (with line numbers where possible) | What Can Be Directly Concluded | What Remains Uncertain | Suggested Next Inspection Step |
|---|---|---|---|---|
| Framework lineage / repo type | `setup.py`, `requirements.txt`; `twilight/` package; `csrc/` (CUDA via FlashInfer JIT); `flash-topk-attention/` submodule -> https://github.com/tsinghua-ideal/flash-topk-attention.git pinned at `d8803b2` (empty until `git submodule update --init`, then populated) | Observed: standalone Python pkg `twilight` installable via `pip install -e .`. Depends on `torch==2.5.0`, `flashinfer-python==0.2.0.post1`, `transformers==4.45.2`, `flash-attn==2.6.3`, `datasets==3.0.1`. Not a fork of any serving framework. ftka kernels exist upstream but are NOT imported by Twilight runtime (only by `benchmark/efficiency/bench_gemv.py:8-10`). | Why the kernels were built but never wired into the runtime path. | Read `05-flash-topk-attention.md`. |
| Entry points (calibration, inference, eval, benchmark) | `benchmark/LongBench/pred.py`; `benchmark/passkey/passkey.py`; `benchmark/RULER/scripts/`; `benchmark/efficiency/bench_gemv.py`; `twilight/pyimpl/__init__.py` -> `enable_sparse_attention`, `reset_sparse_config` | Observed: no calibration entry — channel configs are imported from external DoubleSparse repo paths hardcoded as `/data/chaofan/DoubleSparse/config/...`. Inference is via HF transformers with a monkey-patched attention forward. Benchmarks cover LongBench, Passkey, RULER, plus an operator-level GEMV bench. | Whether `pred.py` was tested on Llama-3.1-8B or only older Llama-2 sizes. | Read RFE table in README. |
| Channel-config artifact format & generation | `twilight/pyimpl/double_sparse.py:22-37 init_model_channel_config()` reads JSON; configs sourced externally | Observed: same JSON format as Implementations A and B (keys = `model.layers.<i>.self_attn.<q\|k>_proj`). No generation code in Twilight itself. Configs live at hardcoded `/data/chaofan/...` paths in `benchmark/configs/config_ds_twi.json`. | Whether Twilight verifies config keys match the loaded model. | Force-load a mismatched config and observe behaviour. |
| Label cache representation | `twilight/pyimpl/double_sparse.py:47-122 double_sparse_selector()`; `twilight/kernel/triton/channel.py:11-79 get_label_tensor_kernel`; `twilight/kernel/triton/bgemv_int8.py:12-73` | Observed: label cache shape `[B, H, r]` for Q_Label and `[B*N_CTX, H, r]` for K_Label, fp16 / int8. `r` (heavy channel dim) defaults to 64; quantisation defaults to int8 with per-token min-max scale (`K_Scales[B*N_CTX,H]`). Per-token min/max quant 2-8 bits via `quant_bit` config. | Whether int8 label and 2-bit channel quant are stable across all tested models. | Sweep `quant_bit in {2,4,8}` on LongBench and report deltas. |
| Token-level top-k selection mechanism | `twilight/pyimpl/attention.py:60-339 attention_forward()` orchestrator; selectors: `quest_selector` (`pyimpl/quest.py:17-122`), `double_sparse_selector` (`pyimpl/double_sparse.py:47-122`), `sparq_selector` (`pyimpl/sparq.py:16-59`), oracle top-k, streaming, tidal-decode | Observed: pluggable selectors via `IndexSelectorType` (`twilight/pyimpl/state.py:12-63`). DS path: Triton `get_label_tensor` to compress Q,K, int8 BGEMV (`bgemv_int8`) to get approx scores, then `torch.topk(scores, token_budget)`. QUEST path: max-pool |K| per chunk, then topk chunk -> token indices. | Whether selector composition (e.g. DS then top-p pruner) is tested for accuracy stability. | Sweep selectors x pruners and tabulate quality vs budget. |
| Custom kernels (language, shapes/dtypes) | Twilight-runtime kernels: CUDA `csrc/src/sampling.cu:22-63` top-p mask + Triton `kernel/triton/{channel.py:56-79, bgemv_int8.py:12-73, qk_int8_per_block.py:23-97}`. Sibling ftka library: CUDA `flash-topk-attention/csrc/src/ftka_ops.cu` (raft_topk + batched_sparse_gemv + int8/int4 variants + quest_sparse_gemv); Triton `ftka/triton_ops/{gemv.py, tokens_moving.py}`; TileLang stub (empty `__init__.py`). | Observed: Twilight runtime path uses Triton-only kernels (`channel`, `bgemv_int8`) + a single CUDA pruner mask (CUB-based prefix scan, SM-aware). ftka adds a vendored RAPIDS RAFT radix-select top-k (`csrc/include/raft_topk.cuh:1-22, 984-1010`; one block per row, no host sync — CUDA-graph friendly) and richer sparse-GEMV variants, but **none** are imported by Twilight runtime. Quest_sparse_gemv is misnamed — it computes the Quest per-page upper-bound `Σ max(q·page_max, q·page_min)`, not a sparse GEMV. ftka's `batched_sparse_gemv_int8_k` is also not a drop-in for Twilight's `bgemv_int8` (ftka iterates full head_dim 128; Twilight iterates only the heavy-channel subset r). | Whether `qk_int8_per_block` is reachable from `pred.py` or benchmarks only. | Trace it; read `05-flash-topk-attention.md`. |
| KV cache layout / memory pool | `twilight/pyimpl/attention.py` uses HF `past_key_value` (DynamicCache) directly | Observed: no custom memory pool. Standard HF KV cache. Attention forward is monkey-patched in place. | Memory cost of label cache is non-trivial — does Twilight free per-step? | Add a memory profile. |
| Attention backend / dispatch routing | `twilight/pyimpl/__init__.py:enable_sparse_attention()` monkey-patches `module.forward = types.MethodType(attention_forward, module)` for `LlamaAttention`/`MistralAttention` (lines 484-494 in `attention.py`) | Observed: runtime method-swap on the HF attention module — applied per-layer, gated by `skip_first_two_layers` (default true). | Whether this composes with HF `torch.compile` or `accelerate.dispatch`. | Try `model = torch.compile(model)` after enabling. |
| Prefill vs decode treatment | `twilight/pyimpl/attention.py:60-339`; SnapKV at `pyimpl/snap_kv.py` (prefill compression only) | Observed: prefill is dense flash-attn; decode is masked-full-attention plus selector mask plus pruner mask (not a fused sparse kernel). SnapKV optionally compresses KV at end of prefill. | Whether prefill could itself benefit from sparse-attn — not attempted. | Check paper Twilight section "discussion". |
| Models supported & their wiring | `twilight/pyimpl/attention.py:484-494` patches `LlamaAttention`, `MistralAttention`; README claims tested on Llama-2-7B-Chat-4K, LongChat-7B-v1.5-32K, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct | Observed: LLaMA and Mistral only via monkey-patch. No DeepSeek/GLM/Qwen support. | Effort to add a new architecture: only need to register a new patch target if the forward signature matches. | Try patching Qwen2Attention as a smoke test. |
| Tests / accuracy validation | Accuracy: LongBench (13 tasks; F1/ROUGE-L), Passkey, RULER. README results: Full(32k)=36.78; Quest(8192) 37.10 -> +Twilight 38.04; DS(8192) 36.62 -> +Twilight 38.71 | Observed: accuracy validation by full-pipeline benchmarks; unit tests embedded as `test_*` functions inside kernel `.py` files. No `pytest` rig. | Whether the README LongBench scores are reproducible without `/data/chaofan/...` paths. | Patch the configs to use relative paths and re-run on a small task. |
| Benchmarks (operator-level and end-to-end) | `benchmark/efficiency/bench_gemv.py` (int8 BGEMV vs torch); LongBench + Passkey + RULER for accuracy | Observed: one operator-level efficiency bench (BGEMV). End-to-end latency/throughput numbers not committed. | No serving-throughput bench. | Wrap a vllm/sglang server around Twilight's HF model. |
| Reproducibility (pinned deps, seeds, datasets, configs) | `requirements.txt` (pinned); `benchmark/configs/*.json`; configs reference `/data/chaofan/...` | Observed: deps tightly pinned. Config paths hardcoded to a single user's filesystem. Calibration delegated to external DoubleSparse repo (no version pinned). | What dataset/seed used for paper numbers — undocumented. | Read the Twilight paper appendix. |
| Hardware assumptions (CUDA, SM, etc.) | `csrc/src/sampling.cu` SM-aware kernel dispatch (line 22-63) | Observed: compute capability >= 7.0 minimum; SM 8.x/9.0 optimised. CUDA version not pinned; FlashInfer + flash-attn imply CUDA 12.0+. Single-GPU. | Tested GPU model unrecorded. | Check `benchmark/README.md`. |
| CPU offloading / multi-tier memory | None. SnapKV (`pyimpl/snap_kv.py`) does prefill-stage KV compression only | Observed: no offload tier. SnapKV only reduces KV size by dropping tokens at end of prefill. | Could Twilight integrate with SGLang HiCache for tiered storage? Open. | Design exercise. |
| Multi-GPU / TP / PP support | None; HF transformers with monkey-patch only | Observed: single-GPU only; no `torch.distributed` use. | Whether the monkey-patch is compatible with HF `device_map='balanced'`. | Try it on 2 GPUs. |
| CUDA-graph compatibility | Not used; HF transformers eager mode. Selector and pruner produce dynamic-shape masks per step | Observed: dynamic masks per step; no `CUDAGraph`. | Could the pruner mask be padded to a fixed size to enable graphs? | Sketch a fixed-budget variant. |
| Quantization integration | Channel quantisation 2-8 bits (`pyimpl/double_sparse.py` `quant_bit`); weight-estimator quant 4-bit typical (`pyimpl/quantize.py:4-13 min_max_per_token_quant_kv`, lines 16-25 `max_per_token_quant_kv`) | Observed: per-token min-max quant for KV scoring; configurable bit width. NO model-weight quantisation (full fp16 weights). | Stability of 2-bit selector quant on long contexts unmeasured. | Sweep on Passkey. |
| Relation to the reference paper | The Twilight paper (cited in repo README) frames itself as a **generalisation** of double-sparsity (the 2408.07092 idea) with an added pruner stage | Observed: not a direct re-implementation of paper 2408.07092 — it's a *framework* in which DS is one of seven selectors, and the novelty is the `weight_pruner` (top-p or threshold) AND-gated with the selector mask. Channel calibration outsourced. Fused flash-topk-attention kernels exist in sibling repo (now fetched) but are NOT wired into Twilight runtime. | The headline LongBench/RULER/Passkey accuracy numbers were produced with the un-fused path because ftka is never imported by the runtime. | See `05-flash-topk-attention.md`. |
| Notable absences / unfinished pieces | `flash-topk-attention/` submodule pinned but kernels orphaned (never imported by runtime); TileLang backend is an empty file (`ftka/tilelang_ops/__init__.py`, 0 lines) despite README advertising it; hardcoded `/data/chaofan/...` paths; no calibration code; no TP/PP; no SGLang/vLLM integration; no speculative-decoding interaction | Observed: significant unfinished engineering — but the research contribution (pruner stage) is reproducible. The kernel library exists but is detached from the inference path; the most reusable piece for an sglang-DS port is `raft_topk` (cuda-graph-friendly, replaces B's blocking host-side `torch.topk`). | Why the kernels were built but not wired in. | See `05-flash-topk-attention.md`. |

---

## Cross-cutting Gaps

These topics are inadequately addressed by **none** or **at most one** of
the three implementations, and are therefore the highest-risk areas for
the port-to-modern-SGLang effort.

1. **MLA-style attention (DeepSeek-V3/V3.2, MiniMax M2, GLM-5.1 variants).**
   None of A/B/C supports MLA. Implementation B's kernel head-dim table
   (16/32/64/128 for sparse-decode; 192/256/288/576 only for the *dense*
   extend kernel) shows the sparse-decode path cannot service MLA's 576
   head dim without new kernels. Channel-config naming
   (`q_proj`/`k_proj`) does not correspond to MLA's compressed-latent
   layout (`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`).
2. **Dynamic `heavy_token_num` per request.** Paper section 4.3 proposes
   this; none of A/B/C implements it. B explicitly comments that
   `heavy_token_num` is static per launch. The per-batch homogeneity
   constraint in B (line 211-214) actively prevents heterogeneous mixed
   short/long decode batches from using sparsity.
3. **CUDA-graph compatibility.** B explicitly disables CUDA graphs
   (`model_runner.py:927-932`). A doesn't use them. C operates in HF
   eager mode. Re-enabling graphs requires removing the host-side
   `torch.topk` round-trip in B's stage 2.
4. **KV-quantisation interaction (fp8/int8 KV).** None of A/B/C
   integrates with fp8 or int8 KV cache. Implementation A has int8
   BGEMV kernels for the label cache, but only for the label path, not
   for the full K/V. Modern SGLang supports `--kv-cache-dtype fp8_e5m2`
   and DS would need to either co-quantise the label cache or accept
   mixed precision.
5. **Tensor parallelism / pipeline parallelism.** A is single-GPU only.
   B's backend is "probably transparent" to TP because heads are
   per-rank but no test exercises it; channel-config head-ordering vs
   TP head-partition is unverified. C is single-GPU only.
6. **Speculative decoding interaction.** None of A/B/C documents or
   tests interaction with EAGLE/Medusa/MTP. SGLang's main spec-decoding
   path lives elsewhere; the DS backend would need to handle draft and
   verify steps coherently.
7. **CPU/multi-tier offloading.** Only A's offloading variant exists,
   and its async-ness is uncertain (DGL `gather_pinned_tensor_rows`
   may be blocking). B has no offload. C has no offload. Integration
   with SGLang HiCache is unexplored.
8. **Numerical equivalence tests.** None of A/B/C asserts that the
   sparse path equals the dense path when `heavy_token_num >=
   max_seq_len`. This would be the cheapest possible correctness
   safety net.
9. **Per-request (not per-batch) routing.** Only B implements routing,
   and it is per-batch, defeating sparsity on mixed-length workloads.
10. **End-to-end serving benchmarks.** A has only operator-level
    attention benchmarks. B has no DS-specific benchmarks at all.
    C has only an operator-level BGEMV benchmark. Throughput, TTFT,
    ITL, and peak-memory deltas vs dense are nowhere reported in code.
11. **Calibration reproducibility.** A's `offline_calibration.py` does
    not pin a seed for the Pile-val sample selection. B and C consume
    the JSON without a format/version check. Config drift across
    re-calibrations is silent.
12. **Non-LLaMA projection naming.** All three assume
    `model.layers.<i>.self_attn.<q|k>_proj` keys. DeepSeek-V3 (MLA),
    GLM-5.1, and Qwen2-MoE will not load these JSONs without a key
    rewrite.

---

### Provenance summary

- This evidence index is a Phase-1 artifact and feeds Phase-2's per-impl
  design documents (`01-doublesparse.md`, `02-sglang-last.md`,
  `03-twilight.md`) and the Phase-3 comparative synthesis
  (`04-comparison-and-recommendations.md`).
- Every row above is sourced from the three Phase-1 survey notes in
  `study/.scratch/survey-*.md` and direct repository inspection.
- Topics where the survey notes themselves flagged uncertainty have been
  retained in the "What Remains Uncertain" column verbatim or with mild
  consolidation; this preserves the audit trail back to the surveys.
