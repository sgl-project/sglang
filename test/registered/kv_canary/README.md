# kv_canary — srt integration tests

Self-unit / self-e2e / self-bench tests for the KV cache canary integration layer
in `python/sglang/srt/kv_canary/`. Auto-discovered by `test/run_suite.py`
via the `register_cuda_ci(...)` decorator in each test file; all files are tagged
`stage="extra-a"`, `runner_config="1-gpu-large"`, plus the self-bench is also
registered to the nightly `nightly-kernel-1-gpu` suite for weekly performance
gating.

The single source of truth for the file list, per-case names, and assertions is
[`lab/docs/pkgs/sglang/notes/2026-05-18-testing/source-of-truth/testing.md`](https://github.com/fzyzcjy/lab) §3
(internal). Do not add, rename, or remove cases here without first updating that
document.
