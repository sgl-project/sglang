# mock_model — pseudo-mode + e2e tests

Self-unit / self-e2e / e2e tests built on top of the mock-engine + oracle
sampler-override harness (see `python/sglang/srt/kv_cache_canary/pseudo_*` and the
sub-design at `lab/docs/pkgs/sglang/notes/2026-05-18-testing/pseudo-model-based-testing/`).
Auto-discovered by `test/run_suite.py` via the `register_cuda_ci(...)` decorator
in each test file; all files are tagged `stage="extra-a"`,
`runner_config="1-gpu-large"`.

The single source of truth for the file list, per-case names, and assertions is
[`lab/docs/pkgs/sglang/notes/2026-05-18-testing/source-of-truth/testing.md`](https://github.com/fzyzcjy/lab) §4
(internal). Do not add, rename, or remove cases here without first updating that
document.
