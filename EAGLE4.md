# EAGLE-4 integration

This checkout supports the external `eagle-harness==0.1.1` SGLang plugin.
Select **`EAGLE_HARNESS`**, the upstream algorithm name. The plugin owns the
EAGLE-4 model, low-rank candidate head, dynamic tree, cost policy and draft/verify
kernels. SGLang owns scheduling, allocation, target execution and state commit.
The EAGLE-4 package is optional and is not copied into SGLang.

## Install and launch (NVIDIA CUDA, Linux x86-64)

Use a Python 3.12 environment with the CUDA toolkit/compiler required by the
current SGLang dependencies. EAGLE-4's published CUDA recipe uses CUDA 13 and
supports NVIDIA SM80 or newer. Install **this source checkout**, not the patched
SGLang wheel from the EAGLE-4 wheel index:

```bash
cd /Users/xk/codex/sglang  # replace with your checkout path on the GPU host
python -m pip install -e './python[eagle4]'
```

To use the supplied EAGLE-4 source instead of the PyPI package, install it in the
same environment after installing SGLang:

```bash
python -m pip install --no-deps -e /Users/xk/codex/EAGLE-4
```

Do not install `EAGLE-4[sglang]` with an engine wheel over this checkout. Verify
which host is imported and check the plugin contract before launching:

```bash
python -c 'import sglang; print(sglang.__file__)'
eagle-harness-check
python -m pytest test/registered/unit/spec/test_eagle4_plugin_contract.py -v
```

If `SGLANG_PLUGINS` is set, include `eagle_harness` in its comma-separated list.
The plugin is discovered through its `sglang.srt.plugins` entry point in each
worker process. A host-hook mismatch refuses plugin registration.

Start with a fixed tree to separate correctness checks from cost-policy tuning:

```bash
sglang serve --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-steps 12 --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode":"static"}' \
  --attention-backend flashinfer \
  --speculative-draft-attention-backend flashinfer \
  --page-size 1 --dtype bfloat16 --disable-overlap-schedule \
  --disable-radix-cache --max-running-requests 8 \
  --cuda-graph-max-bs-decode 8 --mem-fraction-static 0.75
```

Use `--speculative-algorithm-config '{"mode":"serve"}'` for dynamic trees with
live cost measurement. Warm up before benchmarking: cold compilation and cost
collection are not steady-state latency. Omit `--disable-overlap-schedule` to
exercise overlap scheduling. Add `--tp-size N` when the target requires tensor
parallelism. The target and draft must be a matching trained pair.

## What the host provides

* Plugin configuration, request validation, auxiliary target hidden states,
  draft-worker construction and the EAGLE scheduler relay contract.
* CUDA graphs keyed by batch size and verify width, including greedy token-only
  graphs. FlashInfer wrappers are also keyed by width and variant; a missing
  graph runs eagerly rather than replaying an incompatible shape.
* Vocab-parallel greedy argmax and a guarded fused-accept callback. The callback
  cannot bypass grammar, logprobs, penalties, watermarking or sparse KV handling.
* GDN and Mamba2 verify extensions, variable-width state snapshots and accepted
  path state commit. Draft KV capacity uses the draft's own heads, dimensions
  and KV dtype rather than assuming the target's geometry.
* Joint top-k/top-p filtering and min-p filtering in target-only verification.
  Requests with history-dependent penalties commit only the root draw while
  the penalty is active, so subsequent draws use the updated history.

The source hooks and regression tests were selectively ported from the
[official SGLang fork](https://github.com/eagle-hq/sglang), commit
`309f92a7f3afc68fe9e3c7200e0408ef56696b90`, relative to `8e7601ae`.
The plugin source inspected was EAGLE-4 commit
`f71e245a45acc16fd4ebf2b58ce59eba9b2738bf`.
Host changes retain SGLang's license; the separately installed EAGLE-4 package
retains its own license and notices.

## Validation on a GPU host

The manual regression matrix starts a native target server, stops it, and then
starts EAGLE-4. It checks greedy output equality for static, dynamic, overlap
and eager modes, plus logprob and penalty behavior on the static path:

```bash
export EAGLE4_TARGET_MODEL=Qwen/Qwen3.8-27B
export EAGLE4_DRAFT_MODEL=EAGLE-Corp/Qwen3.8-27B-EAGLE4
python -m pytest test/manual/spec/eagle/test_eagle4.py -v
```

Run the host hook tests, including device kernels, in the installed SGLang test
environment:

```bash
python -m pytest -q \
  test/registered/unit/spec/test_spec_registry.py \
  test/registered/unit/spec/test_plugin_hook_signatures.py \
  test/registered/unit/spec/test_eagle_verify_accept_step.py \
  test/registered/unit/spec/test_eagle_verify_penalty_history.py \
  test/registered/unit/spec/test_eagle_verify_target_probs.py \
  test/registered/unit/spec/test_eagle_worker_v2_draft_factory.py \
  test/registered/unit/model_executor/runner/test_decode_graph_variants.py \
  test/registered/unit/layers/test_logits_greedy_tokens.py \
  test/registered/unit/layers/attention/test_flashinfer_verify_plan.py \
  test/registered/unit/layers/attention/linear/test_gdn_verify_extension.py \
  test/registered/unit/layers/attention/mamba/test_mamba2_verify_extension.py \
  test/registered/unit/layers/attention/test_gdn_flashinfer_alignment.py \
  test/registered/unit/kernels/test_ssu_dispatch.py \
  test/registered/unit/mem_cache/test_chunked_decode_alloc.py \
  test/registered/kernels/ops/attention/test_gdn_verify_narrow.py \
  test/registered/kernels/ops/attention/test_fused_verify_gdn_beta_rounding.py \
  test/registered/kernels/ops/mamba/test_ssu_verify_snapshot_pitch.py
```

For performance, launch native target-only, static EAGLE-4 and dynamic EAGLE-4
servers sequentially with the same target, dtype, attention backend, prompt
lengths, output limits and concurrency. Run warmup requests first, then use:

```bash
python -m sglang.benchmark.serving --backend sglang \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 1 --output-file eagle4-c1.jsonl
```

Repeat at concurrency 4/8/16/32 and with representative prompts. Record TPOT,
TTFT, throughput, acceptance length and peak memory. Dynamic trees need not
beat native decode at every concurrency or on every target; qualify the actual
model/hardware pair rather than assuming a speedup.

## Limits and local results

Local validation on 2026-10-22:

* Upstream EAGLE-4 CPU tests for config, limits, cost policies, live costs,
  chain widths and model structure: **303 passed, 1 skipped**.
* Host CPU unit tests executed from the actual source with unavailable runtime
  imports/device boundaries isolated: **63 passed**, including 36 subtests.
  These are method-level tests, not a successful full SGLang import or server
  launch. Four server-resolution/relay cases were excluded from this isolated
  run and remain in the normal test suite.
* **27** target-distribution comparisons against the native PyTorch sampler's
  weights, with top-k/top-p/min-p combinations and mixed request settings.
* The plugin's **38** hook declarations were checked against host source
  signatures with no required mismatch. This is a source contract check;
  `eagle-harness-check` must still run in the accelerator environment.

* The development host is macOS x86-64, with no CUDA or NPU device. Local CPU
  checks and source contracts cannot establish end-to-end model accuracy,
  CUDA-graph correctness, TP synchronization or accelerator performance.
  GPU validation and speedup measurements remain required before deployment.
* CUDA/ROCm are the EAGLE-4 kernel paths. CPU, XPU, MPS and Ascend NPU are
  rejected before worker construction. `sgl-kernel-npu` is unchanged: using its
  EAGLE-3 sampler alone would not implement EAGLE-4 tree drafting or state commit.
  ROCm requires a separate matching ROCm dependency environment and is not
  qualified by the NVIDIA launch example above.
* Text requests only; custom logit processors and watermarking requests are
  rejected at admission because the plugin's fused greedy tail does not apply
  them. Pipeline parallelism is refused by the plugin. Reduced
  draft vocabularies and draft structures the plugin does not implement are
  refused by its checkpoint checks.
* This host uses the plugin's target-only sampling fallback. The optional
  `sampling_step` hook for recorded-q sampled trees is not implemented here.
* Offline chain calibration with native no-spec graph variants is not enabled:
  this host's graph variants vary width, not forward mode. Use tree `static` or
  live-cost tree `serve` as above; the plugin rejects configurations requiring
  the absent forward-mode contract.

See the upstream [installation](https://github.com/eagle-hq/EAGLE-4/blob/main/docs/install.md),
[SGLang guide](https://github.com/eagle-hq/EAGLE-4/blob/main/docs/quickstart-sglang.md)
and [known limits](https://github.com/eagle-hq/EAGLE-4/blob/main/docs/known-limits.md)
for plugin-specific configuration and supported draft structures.
