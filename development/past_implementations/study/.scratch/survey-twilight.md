# Implementation C: Twilight — Survey Findings

## 1. Repo Map
- Top-level: `twilight/` (Python pkg: `pyimpl/` pure-Python selectors/pruners + `kernel/cuda/` + `kernel/triton/`), `csrc/` (CUDA: `src/sampling.cu`, `include/sampling.cuh`, pybind), `flash-topk-attention/` (**empty git submodule** pointing to https://github.com/tsinghua-ideal/flash-topk-attention.git), `benchmark/` (LongBench, RULER, Passkey, efficiency), `figures/`, `assets/`, `setup.py`, `requirements.txt`.
- `pip install -e .` installs the `twilight` Python package (no compiled extension; FlashInfer JIT for CUDA).
- Deps: torch==2.5.0, flashinfer-python==0.2.0.post1, transformers==4.45.2, flash-attn==2.6.3, datasets==3.0.1.

## 2. Entry Points
- LongBench: `benchmark/LongBench/pred.py --model <name> --model_path <path> --task <task> --algo-config-path <cfg.json>`.
- Passkey: `benchmark/passkey/passkey.py`.
- RULER: `benchmark/RULER/scripts/` wrappers.
- Efficiency: `benchmark/efficiency/bench_gemv.py` — int8 BGEMV vs torch.
- **No training/calibration scripts** in repo — channel-config calibration is delegated to external DoubleSparse repo (configs hardcoded under `/data/chaofan/DoubleSparse/config/…`).

## 3. Models Supported
- LLaMA (`LlamaAttention`), Mistral (`MistralAttention`) — monkey-patched at runtime by `enable_sparse_attention()` in `twilight/pyimpl/attention.py` (lines 484–494).
- Tested: Llama-2-7B-Chat-4K, LongChat-7B-v1.5-32K, Mistral-7B-Instruct-v0.3, Meta-Llama-3.1-8B-Instruct (per README table).
- No custom modeling files; method swap via `module.forward = types.MethodType(attention_forward, module)`.

## 4. Where Sparsity Lives
### A. Pipeline: Select → Estimate → Prune
Central orchestrator: `twilight/pyimpl/attention.py:attention_forward()` (lines 60–339), with global state `LocalState` enums in `twilight/pyimpl/state.py` (lines 12–63).

### B. Selectors (IndexSelectorType)
1. **QUEST** (`pyimpl/quest.py:17–122`) — max-pool |K| per chunk, topk on chunk scores, expand to token indices. Called at attention.py line 208.
2. **Double Sparse (DS)** (`pyimpl/double_sparse.py:22–122`) — Triton `get_label_tensor()` extracts heavy channels of Q,K; per-token min-max quant 2–8 bits; matmul on compressed dim; topk. Channel config loaded from external JSON. Called at attention.py 233–242.
3. **SparQ** (`pyimpl/sparq.py:16–59`) — top-r |Q| channels, partial matmul. Called at attention.py 228.
4. **Oracle TopK** (`pyimpl/top_k.py:7–16`) — full attn upper bound.
5. **StreamingLLM** (`pyimpl/streaming_llm.py:7–25`) — keep first num_sinks + last token_budget.
6. **TidalDecode** (`pyimpl/tidal_decode.py:6–30`) — reselect mask at designated layers.

### C. Weight Estimators (approximate attention weights)
- `MIN_MAX_QUANT` (`pyimpl/quantize.py:4–13 min_max_per_token_quant_kv()`).
- `MAX_QUANT` (`pyimpl/quantize.py:16–25 max_per_token_quant_kv()`).
- `NONE` (use real scores).
Used at attention.py lines 247–265.

### D. Weight Pruner (Twilight's innovation)
- `TOP_P` (`pyimpl/top_p.py:10–15 top_p_unnormalized()`) — cumulative prob threshold (e.g. 0.85). Backed by CUDA kernel `sampling::TopPReturnMask<T>` in `csrc/include/sampling.cuh:121`, exposed via `csrc/src/sampling.cu:22–63` (`top_p_fp16_return_mask`, `top_p_fp32_return_mask`). Loaded by FlashInfer JIT in `twilight/kernel/cuda/sampling.py:49–106`.
- `THRESHOLD` (`pyimpl/elementwise_threshold.py:9–18`) — threshold = first attn weight + delta.
- `NONE`.
Mask AND-gating at attention.py 280–281: `mask = mask & pruned_mask`.

### E. Channel calibration
- **Not implemented in Twilight.** Configs come from external DoubleSparse repo.
- `pyimpl/double_sparse.py:22–37 init_model_channel_config()` reads JSON, stores per-head `module.sorted_channel`.

### F. Offloading
- No CPU offload. SnapKV (`pyimpl/snap_kv.py`) does prefill-stage KV compression only.

## 5. Configuration Surfaces (JSON config blocks)
| Block.field | Values | Meaning |
|---|---|---|
| `compressor.type` | `none`, `snap_kv` | Prefill KV compression |
| `selector.type` | `none`/`streaming`/`quest`/`oracle_topk`/`tidal_decode`/`sparq`/`ds` | Token selector |
| `selector.token_budget` | int (e.g. 8192, 1024) | Max tokens after first selector |
| `selector.chunk_size` | int (16 for Quest) | Quest chunk size |
| `selector.r` | int (64 for DS) | Heavy channel dim |
| `selector.quant_bit` | int (2 for DS) | Quant bits for channel scores |
| `selector.config_path` | str | DS external channel config path |
| `selector.selected_channel` | `q`/`k` | DS: which proj to calibrate |
| `weight_estimator.type` | `none`/`min_max_quant`/`max_quant` | Score approx |
| `weight_estimator.quant_bit` | 4 typical | Quant bits |
| `weight_pruner.type` | `none`/`threshold`/`top_p` | **Pruner** |
| `weight_pruner.threshold` | 0.85 typical | Pruner threshold |
| `skip_first_two_layers` | bool (default true) | Skip sparsity in first 2 layers |
| `use_estimated_weights_in_attn` | bool (default false) | Use approx weights in final attn |
- Configs in `benchmark/configs/`: `config_quest_twi.json`, `config_ds_twi.json`, and non-Twilight baselines.

## 6. Custom Kernels
### CUDA (csrc/)
- `top_p_fp16_return_mask`, `top_p_fp32_return_mask` (`csrc/src/sampling.cu:22–42, 45–63`) → CUDA template `sampling::TopPReturnMask<T>` in `csrc/include/sampling.cuh:121`. CUB-based prefix scan; returns bool mask; SM-aware (≥8 → 1024 threads, else 512).
- Loaded via FlashInfer `load_cuda_ops()` (`twilight/kernel/cuda/sampling.py:49–59`); registered as custom torch op (62–106).

### Triton (twilight/kernel/triton/)
- `channel.py:11–79 get_label_tensor_kernel` — gather HEAVY_CHANNEL_NUM cols per head; X[L,H,D], channel[H,r], Out[L,H,r], fp32.
- `bgemv_int8.py:12–73 bgemv_int8_kernel` — Q_Label[B,H,r] × K_Label[B*N_CTX,H,r]ᵀ × K_Scales[B*N_CTX,H]; Out[B,H,N_CTX]; fp16.
- `qk_int8_per_block.py:23–97 _attn_fwd` — fused attn with per-block int8 K (benchmark only, not in main path).

### Flash-TopK-Attention
- Submodule registered (`.gitmodules`) but **empty** in this snapshot. README "(Stay Tuned)".

## 7. Tests / Benchmarks / Reproducibility
- Accuracy: LongBench (13 tasks; metrics F1/ROUGE-L), Passkey needle-in-haystack, RULER.
- Efficiency: `bench_gemv.py`.
- Unit tests embedded as `test_*` functions in kernel `.py` files.
- README results: Full(32k)=36.78; Quest(8192) 37.10 → +Twilight 38.04; DS(8192) 36.62 → +Twilight 38.71.
- Pinned deps (above). Config paths hardcoded to `/data/chaofan/…` — needs local adaptation.

## 8. Hardware Assumptions
- Compute Capability ≥7.0 minimum; SM 8.x/9.0 optimized.
- CUDA version not pinned; constrained by FlashInfer + flash-attn (likely CUDA 12.0+).
- Single-GPU; no TP/PP.

## 9. Important Files & Symbols
1. `twilight/pyimpl/attention.py:60–339` `attention_forward()` — orchestrator.
2. `twilight/pyimpl/state.py:12–63` — type enums.
3. `twilight/pyimpl/double_sparse.py:47–122 double_sparse_selector()`.
4. `twilight/pyimpl/quest.py:17–122 quest_selector()`.
5. `twilight/pyimpl/sparq.py:16–59 sparq_selector()`.
6. `twilight/pyimpl/top_p.py:10–15 top_p_unnormalized()`.
7. `twilight/pyimpl/elementwise_threshold.py:9–18`.
8. `csrc/src/sampling.cu:22–63`.
9. `csrc/include/sampling.cuh:121 TopPReturnMask<T>`.
10. `twilight/kernel/triton/channel.py:56–79 get_label_tensor()`.
11. `twilight/pyimpl/__init__.py` — exports `enable_sparse_attention`, `reset_sparse_config`.
12. `benchmark/configs/config_quest_twi.json`.
13. `benchmark/LongBench/pred.py`.

## 10. Relation to Paper 2408.07092
- Not a direct DS implementation: it's a **generalized sparse-attention framework** that includes DS as one of several selectors.
- Channel calibration outsourced (no calibration code).
- **Adds** a secondary pruner stage (top-p / threshold) AFTER the selector — the central novelty.
- Promised fused flash-topk-attention kernel **not present** in this snapshot.
- Prefill is full flash-attn; decode is masked-full-attention plus selector mask plus pruner mask (not a fused sparse kernel).

## 11. Open Questions
1. Flash-topk-attention never landed — perf claims rest on un-fused path.
2. Where exactly were channel configs generated? External repo; not reproducible from Twilight alone.
3. Prefill vs decode asymmetry — no joint optimization.
4. Quant precision interactions unexplored (channel quant 2-bit, weight quant 4-bit).
5. Top-p threshold (0.85) tuning sensitivity.
6. No TP/PP support.
7. Speculative-decoding compatibility unknown.
