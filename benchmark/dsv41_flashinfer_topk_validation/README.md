# DeepSeek V4.1 index-layer top-k: validation harness

Helper scripts behind the validation evidence of the PR "[Feat] [NVIDIA] deepseek-v4.1: Honor
--dsa-topk-backend in the index-layer top-k". They are added and reverted in that PR's history,
so they are not part of its diff. They ran unmodified on one 8x B300 node in
`lmsysorg/sglang:nightly-dev-cu13-20261115-7b5fb39b`, from `ROOT=/hai-workspace/dsv41-fi-topk/tools`;
`accuracy_table.py` and `bench_runs.py` ran on copies of the result files.

`ROOT` layout: `sglang/` (the PR tree), `sglang-base/` (`git archive` of the base commit's
`python/` and `test/`), `tools/` (these files), `cache/<tree>/` (JIT caches), `sgl-eval-venv/`
(sgl-eval 0.1.3), and the result folders `logs/`, `bench/`, `e2e/`, `logprob2/`, `trace/`. The
checkpoint is at `/data/models/DeepSeek-V4.1-Flash-1017`. To rerun elsewhere, edit `ROOT` and
`MODEL` in the shell scripts.

| file | role |
|---|---|
| `run_tests.sh <sglang\|sglang-base> <log dir>` | the seven pytest suites, one per GPU, with the tree on `PYTHONPATH` |
| `negative_controls.py` | patches one guarded behavior at a time on the PR tree and checks the guarding test fails |
| `verify_graph_backends.py` | runs the CUDA-graph selector test with FlashInfer's op entry points wrapped, logging which kernel each leg captures |
| `bench_topk.py` | operator benchmark: per shape a CUDA graph of R x [PDL producer -> selection], seeded inputs, one variant per process |
| `run_bench_balanced.sh <result dir> [args]` | one warm-up process, then four runs of every variant in Latin-square order on one GPU |
| `bench_final.sh` | the final invocation: `run_bench_balanced.sh`, `analyze_bench.py`, and the 60-round re-run of any cell over 2% |
| `analyze_bench.py <result dir>` | pooled per-shape medians, per-shape table and suite geomeans |
| `bench_runs.py <result dir>` | per-run ratios, per-slot spread, the fixed per-replay-cost fit |
| `trace_sgl_kernel.py`, `trace_rerun.sh` | base vs branch (and base vs base) kernel launches, output hashes and top-k validity per benchmark shape |
| `run_e2e.sh`, `e2e_queue.sh` | serving runs (B300 High-Throughput recipe, TP4 + EP4) with sgl-eval GSM8K and RULER2 64K, one server per half node |
| `pregen_ruler2.py` | generates the RULER2 64K dataset once for the checkpoint tokenizer |
| `accuracy_table.py <e2e root> <out dir> <runs...>` | scores with standard errors, paired differences, RULER2 subtasks, raw aggregates |
| `run_logprob.sh`, `logprob_rounds2.sh`, `logprob_equivalence.py` | greedy logprob comparison on the RULER2 prompts: collect per server, compare pairs |
