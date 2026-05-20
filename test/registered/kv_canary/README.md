# kv_canary — srt integration tests

Self-unit / self-e2e / self-bench tests for the KV cache canary integration layer
in `python/sglang/srt/kv_canary/`. Auto-discovered by `test/run_suite.py`
via the `register_cuda_ci(...)` decorator in each test file; all files are tagged
`stage="extra-a"`, `runner_config="1-gpu-large"`, plus the self-bench is also
registered to the nightly `nightly-kernel-1-gpu` suite for weekly performance
gating.

See `test/README.md` for how tests are discovered, registered, and run.
