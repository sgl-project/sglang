# DeepSeek V4.1 index-layer top-k: validation harness

Helper scripts behind the validation evidence of the PR "[Feat] [NVIDIA] deepseek-v4.1: Honor
--dsa-topk-backend in the index-layer top-k". They are added and reverted in that PR's history,
so they are not part of its diff. They ran unmodified on one 8x B300 node in
`lmsysorg/sglang:nightly-dev-cu13-20261115-7b5fb39b`, from `ROOT=/hai-workspace/dsv41-fi-topk/tools`;
`accuracy_table.py` and `bench_runs.py` ran on copies of the result files. The operator
benchmark (`final-d`) ran on a second node with `lmsysorg/sglang:nightly-dev-cu13-20261116-0b098e12`
(same library versions), after `prewarm_jit.sh`. This version supersedes the harness commit
`b0816a8`; `results/` holds raw outputs too long for the PR body.

`ROOT` layout: `sglang/` (the PR tree), `sglang-base/` (`git archive` of the base commit's
`python/` and `test/`), `tools/` (these files), `cache/<tree>/` (JIT caches), `sgl-eval-venv/`
(sgl-eval 0.1.3), and the result folders `logs/`, `bench/`, `e2e/`, `logprob2/`, `trace/`. The
checkpoint is at `/data/models/DeepSeek-V4.1-Flash-1017`. To rerun elsewhere, edit `ROOT` and
`MODEL` in the shell scripts.

| file | role |
|---|---|
| `run_tests.sh <sglang\|sglang-base> <log dir>` | the seven pytest suites, one per GPU, with the tree on `PYTHONPATH` |
| `negative_controls.py` | patches one guarded behavior at a time on the PR tree and checks the guarding test fails |
| `mode2_controls.py <tree>` | breaks the mode-2 parity check's premises (sgl-kernel top-k v2 built in mode 1; FlashInfer large tie-break) and checks it fails, then runs it unmodified |
| `verify_graph_backends.py` | runs the CUDA-graph selector test with FlashInfer's op entry points wrapped, logging which kernel each leg captures |
| `bench_topk.py` | operator benchmark: per shape a CUDA graph of R x [PDL producer -> selection], seeded inputs, one variant per process; variants base, branch, mode 2 (`SGLANG_TOPK_V2_MODE=2`), FlashInfer fused, small, small + deterministic, each checking its environment |
| `run_bench_balanced.sh <result dir> [args]` | one warm-up process, then six runs of the six variants in Williams-square order on one GPU |
| `bench_final.sh` | the final invocation (`BENCH_NAME`, default `final-d`): `run_bench_balanced.sh`, `analyze_bench.py`, and the 60-round re-run of any cell over 2% |
| `analyze_bench.py <result dir>` | pooled per-shape medians, per-shape table and suite geomeans |
| `bench_runs.py <result dir>` | per-run ratios, per-slot spread, the fixed per-replay-cost fit (six runs) |
| `trace_sgl_kernel.py`, `trace_rerun.sh` | base vs branch (and base vs base) kernel launches, output hashes and top-k validity per benchmark shape |
| `run_e2e.sh`, `e2e_queue.sh` | serving runs (B300 High-Throughput recipe, TP4 + EP4) with sgl-eval GSM8K and RULER2 64K, one server per half node |
| `pregen_ruler2.py` | generates the RULER2 64K dataset once for the checkpoint tokenizer |
| `accuracy_table.py <e2e root> <out dir> <runs...>` | scores with standard errors, paired differences, RULER2 subtasks, raw aggregates |
| `run_logprob.sh`, `logprob_rounds2.sh`, `logprob_equivalence.py` | greedy logprob comparison on the RULER2 prompts: collect per server, compare pairs |
| `prewarm_jit.sh <scratch dir>` | one throwaway pass of every variant to fill a fresh node's JIT caches before a benchmark |
| `topk_order_check.py <selector test file>` | per `SGLANG_TOPK_V2_MODE`: whether repeated sgl-kernel top-k v2 calls return the same set and the same output order |

| `results/` file | content |
|---|---|
| `accuracy-details.md` | RULER2 subtasks, every serving run's `run.txt`, sgl-eval settings and raw `aggregate` block |
| `trace-compare.md` | `trace_sgl_kernel.py compare` output with the full kernel names |
| `final-d-analysis.md` | `analyze_bench.py` output of the final benchmark (3-decimal us, per-variant environment) |
| `final-d-runs.md` | `bench_runs.py` output of the final benchmark |
| `logprob2-report.md` | `logprob_equivalence.py compare` output (table and per-pair JSON) |
| `topk-order-check.log` | `topk_order_check.py` output in modes 2 and 1 |
