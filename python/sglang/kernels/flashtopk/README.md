# FlashTopK (vendored)

[FlashTopK](https://github.com/Tencent-Hunyuan/FlashTopK) at commit
`090902abad86e5bb3324519124a680122fff91a1` (`main`), Apache-2.0 (`LICENSE`, `NOTICE`).
It backs `SGLANG_OPT_USE_PREFILL_FLASHTOPK=1` through
`sglang/kernels/ops/attention/dsa/flashtopk_prefill.py`.

The `flashtopk/` package is copied verbatim from upstream except for two local
edits:

- Its custom op is registered as `torch.ops.sglang_flashtopk.select` (`ops.py`,
  `__init__.py`), so the vendored copy can be imported alongside a separately
  installed `flashtopk`.
- Caller-supplied scratch (`topk.py`, `plan.py`, `recovery.py`,
  `kernels/launch.py`): `Selector.scratch_bytes(num_q, seq_len_kv)` sizes a
  call's scratch, `Selector.plan(..., scratch=False)` leaves scratch out of the
  persistent plan, and `Selector.run(..., scratch=buffer)` carves candidates,
  slice scores, recovery staging, and pruning plan/mask out of `buffer`.
  SGLang allocates the scratch transiently per prefill call so FlashTopK holds
  negligible permanent GPU memory and does not reduce the KV cache pool.

Runtime dependencies are already pinned in `python/pyproject.toml`: `torch`,
`triton`, `nvidia-cutlass-dsl`, and `apache-tvm-ffi`. The CuTe DSL and Triton
kernels compile at runtime (pre-compiled during `ModelRunner` startup warmup).

To re-sync, copy `flashtopk/`, `LICENSE`, and `NOTICE` from the new upstream
commit, reapply the two edits above, and update the commit hash here. The
pre-commit config excludes this directory so formatting never diverges from
upstream.
