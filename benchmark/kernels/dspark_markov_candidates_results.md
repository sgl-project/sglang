# DSpark Markov candidates: reproduction and validation record

This record distinguishes implemented checks from completed GPU validation.
It contains no measured GPU speedup or user-checkpoint acceptance claim.

## Local environment and status

- SGLang baseline: `d6090f92bf60dd6d0a02d8c02ec535f6f66ab1a9`.
- Development branch: `feat/topk-markov`; obtain the final revision with `git rev-parse HEAD` and preserve `git diff` when measuring uncommitted changes.
- Local host: macOS ARM64; no NVIDIA CUDA device. The requested `/personal/model/Qwen3-4B` and `/personal/checkpoints/speculators/dspark_qwen3_4b_redhat/checkpoints/0` are unavailable here.
- Local CPU environment: macOS 15.5 ARM64, Python 3.12.0, Torch 2.13.0 (`torch.version.cuda=None`, `torch.cuda.is_available()=False`), Transformers 5.14.1, msgspec 0.21.1. No Triton or FlashInfer distribution is installed; any platform import stubs do not compile or execute GPU kernels.
- GPU, driver, CUDA, Triton/FlashInfer execution versions, checkpoint weight shapes, DP8 startup pressure, and model metrics are **not measured**. The benchmark emits installed runtime versions, device identity, arguments, timing method, actual candidate dispatch path, and memory accounting on its execution host.
- Static compilation of the changed Python files is completed. The reproducible isolated checkpoint command below passed 22 tests; the isolated candidate command passed 10 tests, including weight refresh and CPU noise statistics. These use real production functions with isolated imports and do not validate full runtime imports. Normal pytest collection currently fails when Torch/Torchvision imports the existing macOS Triton stub's missing `triton.backends`; this is a local runtime dependency/platform limitation, not a passing test result. CUDA tests, runtime graph tests and the opt-in user-checkpoint test have **not run**. CUDA skips do not count as GPU passes.

## Implemented scope and bounds

The candidate dispatch targets Qwen3 vanilla Markov heads with directly readable FP16/BF16/FP32 weights on NVIDIA CUDA at TP=1. Independent DP workers each own their table and cache; DP1/DP8 execution still needs server validation. Quantization elsewhere in the model does not disable this dispatch. The initial resource bounds are K<=64, M<=128, rank<=1024 and gamma<=32, with contiguous rank dimensions and nonoverlapping weight rows. These bounds prevent unbounded specializations; they are **not GPU-profiled capacity guarantees**. K=0, gated/RNN heads, unsupported TP/device/weights or resource bounds select the existing dense proposal and log why. Invalid checkpoint shapes, missing weights and invalid mappings remain loading errors.

K resolves from the new CLI override, checkpoint `markov_topk`, checkpoint `dspark_draft_topk`, then 0. M resolves from its CLI override, checkpoint `markov_bias_topk`, then 16; effective M is 0 when K is 0. Explicit zero overrides checkpoint values. Each budget must fit Vd, while their sum may exceed Vd. `speculative_eagle_topk` remains 1.

Speculators normalization supports the Qwen3 dense backbone with auxiliary target hidden states. It preserves independent input embeddings/LM heads and validates injective draft-to-target offsets. Omni, other backbones and `use_aux_hidden_state=False` are rejected explicitly. The unavailable user checkpoint's actual schema and weights have not been verified against this adapter. Weight reloads rebuild static tables in place; changed layouts, mappings or scale require worker restart and graph recapture.

## Correctness checks

Use the project's uv-managed environment on the execution host:

```bash
# These two commands were actually run on the local CPU-only host.
.venv/bin/python test/registered/unit/spec/run_dspark_checkpoint_isolated.py
.venv/bin/python benchmark/kernels/check_dspark_markov_candidates_cpu.py

# Requires a complete supported SGLang runtime; currently blocked locally.
PYTHONPATH=python .venv/bin/python -m pytest -q \
  test/registered/unit/spec/test_dspark_checkpoint.py \
  test/registered/spec/dspark/test_dspark_markov_candidates.py \
  test/registered/spec/dspark/test_dspark_candidate_runtime.py
PYTHONPATH=python .venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_kernel_parity.py
```

The candidate suite derives expected candidates and scores from original base logits and W1/W2 with scalar arithmetic. Exactly representable weights isolate semantic errors. It checks unions, overlap, complete vocabulary coverage, negative/zero scales, odd dimensions/rank, nonidentity mapping, mixed greedy/sampling, and actual predecessor chains. Candidate-boundary ties are avoided except the explicitly defined zero-scale static ranking; greedy score ties use the smallest target token ID.

CUDA tests run the actual production sampler and SGLang's actual chain rejection kernel. Proposal tests use 131,072 independent random seed draws per dtype and nonunit temperature. A predetermined two-sided Bernstein bound with union correction limits the family-wise false rejection probability to `1e-6`; do not rerun a statistical failure until it happens to pass. The rejection test independently enumerates the small-vocabulary, two-step proposal/accept/reject/bonus branches, includes target probability outside proposal support, and must reject a deliberately corrupted q that doubles one token's weight. Proposal, acceptance and final/residual draws use separate RNG streams.

The existing DSpark sampling path does not consume `sampling_info.sampling_seed`. The candidate path preserves advancement of Torch's global CUDA generator, drawing fresh `[B,N]` seeds on each launch/replay and keying candidate noise by token identity. This guarantees stability under candidate-slot reordering for a fixed seed; it does not establish per-request reproducibility across batch reordering. The verifier retains its separate subsequent Torch RNG draws. GPU distribution and replay behavior remain unverified locally.

Additional CUDA tests cover cache slot reuse/reordering, smaller batches, separate capacities, graph/eager equality with explicit seeds, graph input updates and fresh RNG on replay. The runtime suite exercises the real `DsparkDraftSampler` and eager wrapper with a tiny known-logit backbone, including both anchor conventions, mixed sampling, graph valid-row staging, cache ownership, confidence predecessor input, and unsupported-path dispatch. These CUDA tests are authored but **not run locally**. Full server request/position staging, CUDA sanitizer, confidence calibration, multi-process TP execution, DP8 routing, target greedy parity and model quality remain server validation tasks.

```bash
# Run on NVIDIA CUDA after the ordinary correctness suite passes.
compute-sanitizer --tool memcheck .venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_markov_candidates.py \
  -k 'cache_reorder or actual_kernel_chain or graph_and_eager'
compute-sanitizer --tool racecheck .venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_markov_candidates.py \
  -k 'cache_reorder or graph_and_eager'
```

## Microbenchmark

First inspect the actual checkpoint and set `VD` and `RANK` from its W2 shape; do not infer either from the target model name. Run correctness before timing. Example commands intentionally require these values rather than inventing checkpoint metadata:

```bash
VD=<actual-draft-output-vocabulary>
RANK=<actual-markov-rank>
PYTHONPATH=python .venv/bin/python benchmark/kernels/bench_dspark_markov_candidates.py \
  --vocab "$VD" --rank "$RANK" --dtype bfloat16 --base-dtype float32 \
  --k 32 --m 128 --gamma 8 --batch 1 4 16 32 64 128 \
  --sampling probabilistic --temperature 0.8 --cache cold \
  > dspark-k32-m128-eager-cold.jsonl
PYTHONPATH=python .venv/bin/python benchmark/kernels/bench_dspark_markov_candidates.py \
  --vocab "$VD" --rank "$RANK" --dtype bfloat16 --base-dtype float32 \
  --k 32 --m 128 --gamma 8 --batch 1 4 16 32 64 128 \
  --sampling probabilistic --temperature 0.8 --cache hot --graph \
  > dspark-k32-m128-graph-hot.jsonl
```

Repeat with `--sampling greedy` and `--sampling mixed`, matching the checkpoint's actual dtype and logit scale (`--alpha`). Sweep `(K,M)=(32,0),(32,16),(32,64),(64,128)` and `--gamma 1/4/8`. Keep parameter changes separate from implementation changes.

The microbenchmark compares the existing dense sampling kernel (or existing `MarkovGreedyStep` for greedy), a staged Torch candidate implementation, and the production candidate wrapper. All required RNG generation within those wrappers is timed. Production Top-K, walk plus sparse cache, Torch Top-K, and dense q softmax are separately labeled measurements. The walk ablation excludes RNG and Top-K; do not present its latency as total draft latency or subtract unrelated medians to fabricate isolated phases. The candidate path must report `triton` or the benchmark stops.

Timing prefers FlashInfer CUPTI with an explicit cold/hot policy. If the installed harness cannot provide it, `--timing auto` reports the reason and uses CUDA events; `--timing cupti` fails instead. Warmup/JIT and graph capture precede steady-state timing. Graph capture time and initialization memory are recorded separately. Cold event timing flushes 256 MiB before recording the event. Report p50/p95 and timer, cache assumptions and graph mode with every result. GPU events include launch gaps in eager mode; CUPTI and graph times have different measurement boundaries and must not be silently mixed.

This synthetic microbenchmark starts from base logits. It excludes the backbone/LM head, full-vocabulary TP gather, confidence, target forward and rejection, scheduling and server input staging. Its standalone dense q softmax ablation is not an actual verifier measurement. Profiler kernel names and compiled resource reports are still needed to establish registers/spills, W2 gather scaling with K, per-phase attribution and end-to-end bottlenecks. No automatic claim of a high-performance result follows from this script passing.

## User checkpoint and serving reproduction

Inspect local JSON and safetensors metadata first (uses `safe_open`, reads no weight payloads, downloads nothing, and does not validate mapping values):

```bash
.venv/bin/python benchmark/kernels/inspect_dspark_checkpoint.py \
  /personal/checkpoints/speculators/dspark_qwen3_4b_redhat/checkpoints/0 \
  > dspark-checkpoint-metadata.json
```

The inspector passed a local synthetic shape/dtype and invalid-shard check. The command against the requested checkpoint returned an explicit missing `/personal` error, not a compatibility result.

After checkpoint inspection and single-GPU smoke, the requested server configuration is:

```bash
SGLANG_RAGGED_VERIFY_MODE=static PYTHONPATH=python .venv/bin/python -m sglang.launch_server \
  --model-path /personal/model/Qwen3-4B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /personal/checkpoints/speculators/dspark_qwen3_4b_redhat/checkpoints/0 \
  --speculative-dspark-block-size 8 \
  --speculative-dspark-markov-topk 32 \
  --speculative-dspark-markov-bias-topk 128 \
  --tp-size 1 --dp-size 8 --max-running-requests 128 --disable-radix-cache
```

Use `--dp-size 1` for the initial controlled comparison and explicitly set `--speculative-dspark-markov-topk 0` for the same-branch baseline. Check startup logs for effective K/M, actual head and vocabulary shapes, graph route, fallback reason, table construction and allocated cache. A request with `temperature=0.8, top_k=-1, top_p=1.0` exercises the probabilistic path; use `temperature=0` for greedy. The model's candidate K is independent of request top_k.

The candidate implementation currently requires an explicit `--max-running-requests` to bound persistent proposal memory before KV allocation. Without that bound it reports a full-vocabulary fallback. Capacity includes the largest graph tier, so the actual allocation may exceed active request count. The requested value 128 satisfies this condition.

The opt-in test in the existing basic sanity suite sequentially launches target-only and DSpark K32/M128 servers for eager and graph modes, checks exact greedy text and token parity, sends a real mixed request batch with `temperature=0.8, top_k=-1, top_p=1.0`, and rejects startup logs that report a fallback. It requires existing local checkpoint directories and never substitutes a public checkpoint. Select the class explicitly so the pre-existing public-checkpoint suite is not launched:

```bash
SGLANG_TEST_DSPARK_USER_CHECKPOINT=1 \
  SGLANG_TEST_DSPARK_TARGET_PATH=/personal/model/Qwen3-4B \
  SGLANG_TEST_DSPARK_DRAFT_PATH=/personal/checkpoints/speculators/dspark_qwen3_4b_redhat/checkpoints/0 \
  PYTHONPATH=python .venv/bin/python -m pytest -q -s \
  test/registered/core/test_basic_sanity_dspark.py::TestDSparkUserCheckpointCandidates
```

This opt-in test is **not run locally**. It is a TP1/DP1 serving smoke with gamma 8, max-running-requests 128 and prefix caching disabled. Successful mixed requests are not a statistical distribution test or task-quality evaluation. It has no speed assertion and does not establish DP8 behavior or an end-to-end performance gain.

Record actual schema, architecture, model_type, Speculators wrapper, input/target/draft vocabulary sizes, mask ID, rank/head type, scale, anchor convention, query rows/gamma, layer mapping, confidence config and head/embed/d2t tensor shapes before claiming checkpoint compatibility. No such inspection of the requested checkpoint was possible locally.

For serving measurements fix dataset, request sampling, length policy, arrival load and cache settings; alternate baseline/candidate repetitions. Record client concurrency and each DP replica's observed batch histogram, rather than assuming DP8 implies B128. Report accepted draft tokens excluding bonus, emitted tokens including bonus, acceptance by position, draft/verify/iteration latency, tokens/s, TPOT/ITL p50/p95, memory, initialization and capture. Analyze `E[emitted tokens] / E[iteration time]`. These measurements and Qwen3-4B task quality results remain **unverified**.
