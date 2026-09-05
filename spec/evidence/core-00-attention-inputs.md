Status: implemented

# Attention input regression evidence

CI at `122d66e51229b0f10ca2c8ad2d024df59880e06d` exposed three failures in
code inherited from upstream:

- [NPU vision graph startup](https://github.com/sgl-project/sglang/actions/runs/33958643619/job/101300656909):
  `_get_graph_key()` was called without its required sequence boundaries.
- [Kimi Linear DCP prefill](https://github.com/sgl-project/sglang/actions/runs/33958643633/job/101286617275):
  TokenSpeed dereferenced `rotary_emb.cos_sin_cache` for a model with no rotary
  embedding module.
- [Lean paged attention](https://github.com/sgl-project/sglang/actions/runs/33958643752/job/101289417337):
  the test supplied a four-dimensional KV buffer to the three-dimensional
  slot-buffer interface.

The NPU and TokenSpeed regression tests reproduced their respective exceptions
before the fixes. The NPU test verifies graph selection, workspace shape, and
retained sequence boundaries with mocked device capture. The TokenSpeed test
compares FP8 Q/K/V and latent cache writes with explicit tensor references for
both absent and identity rotary embeddings. The existing Lean test retains its
shuffled slots, page sizes, and numerical parity thresholds.

Run from the repository root with the local environment available:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" CUDA_VISIBLE_DEVICES=0 \
  timeout 300 .venv/bin/python -m pytest -q \
  test/registered/unit/multimodal/test_vit_cuda_graph_runner.py \
  test/registered/attention/unittests/mla/test_tokenspeed_mla.py::TestTokenspeedMLAPrefillPreparation \
  test/registered/kernels/test_lean_attention.py::TestLeanAttentionParity::test_paged_kv_parity \
  --disable-warnings --tb=short
```

Result on 2026-09-05 in the isolated checkout on NVIDIA H200:

```text
9 passed, 15 warnings, 6 subtests passed in 9.42s
```

All applicable pre-commit checks, Ruff lint/format, isort, and `git diff --check`
also passed. Local checks exclude
the uncommitted sampler edit. Live NPU graph execution, Blackwell DCP acceptance,
and MI35x execution require their hardware-specific CI jobs; the local result
does not claim those jobs have passed.
