# MoE prefill CUDA graph validation harness

Scripts that produced the validation evidence for the FlashInfer CuTe DSL MoE and
MegaMoE prefill CUDA graph change. They run the registered e2e classes unchanged;
nothing here is imported by SGLang. The scripts are byte-identical to what ran.

- `run_case.sh <gpus> <run> <test file> <TestClass | all>`: run one registered class (or
  a whole file) on four GPUs from the checkout `$ROOT/$TREE` with its `python/` on
  `PYTHONPATH`, writing `$ROOT/logs/<run>/test.log`. Each run gets a fresh
  `SGLANG_CACHE_DIR`, so FlashInfer autotuning (including the EXTEND pass and MegaMoE
  profiles) starts cold, unless `CACHE_NAME` names a shared cache. JIT caches are shared.
- `queue.sh <gpus> "<run>|<file>|<class>|<tree>|<cache>" ...`: run cases back to back on
  one four-GPU slot; `WAIT_FOR=<run>` delays the first case until that run exits.
- `discovery_cases.py`: subclasses of the registered classes that pin
  `--cuda-graph-backend-prefill breakable` (to bypass auto-disable rules on the base
  commit), and `NemotronFp8DefaultMoeBackend`, a Nemotron-H FP8 GSM8K class on the default
  MoE backend used as a sanity check for the SM100-family hybrid prefill graph change.
- `capture_time_ab.sh <gpus> <run> [rounds]`: launch the CuTe DSL NVFP4 reload-suite server
  with `SGLANG_MEMORY_SAVER_CUDA_GRAPH` = 0 and 1 and record the capture lines.
- `base_behavior_check.py`: with an SGLang tree on `PYTHONPATH`, replay the new
  `TestExtendAutotuneDummy` scenario and print the EXTEND dummy that tree requests for a
  non-speculative and a speculative target (stubbing `get_disagg` where the tree reads it).
- `summarize_runs.py [--logs DIR] [--split] [run ...]`: per run (with `--split`, per server
  launch of a whole-file run), the verdict, GSM8K score, EAGLE
  accept length, target prefill graph memory, prefill batches replayed from a graph versus
  eager (`Prefill batch ... cuda graph: True/False`, summed over DP-rank log lines), the
  EXTEND autotune pass line, and rank-0 prefill capture time/memory.

`ROOT` defaults to the devbox workspace used for the PR; set it to a directory that holds
the checkout(s) and a `logs/` folder. Models are fetched from the Hugging Face Hub.
