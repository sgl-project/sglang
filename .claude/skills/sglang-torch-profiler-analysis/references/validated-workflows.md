# Validated Workflows

These were the concrete workflows used to validate the skill design against real remote GPU environments.

## 1. Small single-GPU Qwen

Validated as a two-pass workflow on H100.

### Pass 1: no-CUDA-graph mapping pre-pass

Validated example:

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_torch_profile_qwen25_15b_map
CUDA_VISIBLE_DEVICES=5 FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 32240 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

Then:

```bash
python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --url http://127.0.0.1:32240 \
  --num-steps 5 \
  --profile-by-stage \
  --profile-prefix qwen25_15b_map \
  --export-kernel-map /tmp/qwen25_15b_kernel_map.json
```

### Pass 2: final optimized profile

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_torch_profile_qwen25_15b_final
CUDA_VISIBLE_DEVICES=5 FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 32241

FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.profiler \
  --url http://127.0.0.1:32241 \
  --num-steps 5 \
  --profile-by-stage \
  --profile-prefix qwen25_15b_final

python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --input /tmp/sglang_torch_profile_qwen25_15b_final \
  --kernel-map /tmp/qwen25_15b_kernel_map.json
```

## 2. Multi-GPU Qwen

Validated as a two-pass workflow on H100 with `Qwen/Qwen3-32B`.

### Pass 1: no-CUDA-graph mapping pre-pass

Validated target options:

- `Qwen/Qwen3-32B` on H100 with `--tp 2`
- `Qwen/Qwen3-Next-80B-A3B-Instruct` on H200 with `--tp 4`

Example:

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_torch_profile_qwen32b_map
CUDA_VISIBLE_DEVICES=1,2 FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32040 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

Then:

```bash
python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --url http://127.0.0.1:32040 \
  --num-steps 5 \
  --profile-by-stage \
  --profile-prefix qwen32b_map \
  --export-kernel-map /tmp/qwen32b_kernel_map.json
```

### Pass 2: final optimized profile

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_torch_profile_qwen32b_final
CUDA_VISIBLE_DEVICES=1,2 FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32041

FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.profiler \
  --url http://127.0.0.1:32041 \
  --num-steps 5 \
  --profile-by-stage \
  --profile-prefix qwen32b_final

python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --input /tmp/sglang_torch_profile_qwen32b_final \
  --kernel-map /tmp/qwen32b_kernel_map.json
```

## Notes

- Prefer `TP-0` traces first for kernel share analysis and for the exported kernel map.
- If the directory only contains merged traces, state that explicitly in the final conclusions.
- For stage-aware comparisons, analyze `EXTEND` and `DECODE` separately before summarizing the overall model behavior.
- On current SGLang builds, add `--disable-piecewise-cuda-graph` together with `--disable-cuda-graph` for the mapping pass, otherwise extend/prefill may still run under piecewise CUDA graph.
- `Qwen/Qwen3-4B-Instruct-2507` was present in H200 cache during validation but did not work on the validated stack because of a `Qwen3Config.rope_parameters` compatibility issue.

## 3. Deliberately broken TP fusion rediscovery

Validated on B200 with `Qwen/Qwen2.5-0.5B-Instruct`, `TP=2`.

The validation intentionally commented out the fused TP all-reduce + RMSNorm path inside:

- `python/sglang/srt/layers/layernorm.py`

and forced the code to fall back to:

- plain `tensor_model_parallel_all_reduce`
- then ordinary `norm_module.forward(...)`

### Mapping pass

Graph-off server:

```bash
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32260 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

Then:

```bash
python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --url http://127.0.0.1:32260 \
  --num-steps 5 \
  --profile-by-stage \
  --profile-prefix qwen25_tp2_map \
  --export-kernel-map /tmp/qwen25_tp2_map_kernel_map.json
```

Observed result:

- the fuse table rediscovered `TP all-reduce + residual/RMSNorm`
- it pointed back to `python/sglang/srt/layers/layernorm.py:89 _forward_with_allreduce_fusion`

### Formal pass

Graph-on server with the intentionally broken code still in place:

```bash
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32261
```

Then:

```bash
python3 scripts/analyze_sglang_torch_profile.py triage \
  --mapping-input /tmp/sglang-torch-profile-_7gd033i/1775035260.3725493 \
  --formal-input /tmp/sglang-torch-profile-64oszwu2/1775035372.2416728
```

Observed result:

- the kernel table showed `void sglang::cross_device_reduce_1stage<__nv_bfloat16, 2>` at roughly `31%` decode share
- the overlap table surfaced that same communication kernel as a top headroom row
- the fuse table again flagged `TP all-reduce + residual/RMSNorm`

This validation demonstrates that the skill can rediscover a real missing fusion path and also surface the exposed overlap opportunity that appears when the fusion is removed.

## 4. Dense Qwen3 QK-norm + RoPE rediscovery

Validated on B200 with `Qwen/Qwen3-32B`, `TP=2`.

### Mapping pass

Graph-off server:

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32320 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

### Formal pass

Graph-on server:

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp 2 \
  --host 127.0.0.1 \
  --port 32321
```

Observed result:

- on the current `sm100 + trtllm_mha` stack, `apply_qk_norm` and `RoPE` may collapse into a shared generic `void` kernel row
- the useful source evidence still survives in the mapped Python locations:
  - `python/sglang/jit_kernel/rope.py:179 apply_rope_with_cos_sin_cache_inplace`
  - `python/sglang/srt/models/utils.py:204 apply_qk_norm`
- the fuse detector therefore needs to treat a shared kernel row containing both `QK norm` and `RoPE` evidence as valid
- after broadening the rule, triage produced:
  - `decode | Q/K RMSNorm + RoPE before attention | Conditional | 2.34 ms | 4.5%`

This validation demonstrates that dense-Qwen3 fuse detection should be source-evidence driven, not tied only to `norm` or `rope` kernel categories.

## 5. Single-GPU negative control

Validated on B200 with `Qwen/Qwen2.5-0.5B-Instruct`, single GPU, `TP=1`.

Observed result:

- the three main tables were still produced normally
- no medium-confidence source-backed fusion opportunity was emitted
- the skill did not incorrectly report:
  - `TP all-reduce + residual/RMSNorm`
  - `Q/K RMSNorm + RoPE before attention`

This validation is the negative control for avoiding TP-specific false positives when no TP communication exists.

## 6. MiniMax overlap-heavy trace validation

Validated on B200 using previously captured `MiniMaxAI/MiniMax-M2.5` mapping and formal traces.

Current note:

- a fresh launch on the current B200 `main` repo failed before profiling with:
  - `AttributeError: 'MiniMaxM2Config' object has no attribute 'rope_theta'`
- that load-time issue is separate from the profiler skill itself

Using the existing B200 traces still validated the triage behavior:

- the kernel table was dominated by:
  - `void sglang::cross_device_reduce_1stage<__half, 4>`
  - `fused_moe_kernel`
- the overlap table stayed compact and preserved only actionable overlap rows plus `low-roi-hidden` deprioritization rows
- the fuse table flagged `TP all-reduce + residual/RMSNorm`

This validation demonstrates that the compact three-table artifact still stays readable on a communication-heavy MoE model.
