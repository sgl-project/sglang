# mock_model — pseudo-mode + e2e tests

Self-unit / self-e2e / e2e tests built on top of the mock-engine + oracle
sampler-override harness (see `python/sglang/srt/kv_canary/pseudo_*`).
Auto-discovered by `test/run_suite.py` via the `register_cuda_ci(...)` decorator
in each test file; all files are tagged `stage="extra-a"`,
`runner_config="1-gpu-large"`.

See `test/README.md` for how tests are discovered, registered, and run.
