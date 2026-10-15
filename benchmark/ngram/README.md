# Isolated code-edit benchmark

`bench_code_edits.py` runs SGLang against the prepared TensorRT-LLM code-edit
workload. It consumes the existing `manifest.json`, `dataset.jsonl`, and pinned
`model/` snapshot from `benchmark/sa_spec/bench.py prepare`. It never downloads
new inputs, adds a chat template, shuffles prompts, or reads reference solutions.
The existing TRT helper is included with repository formatting applied; its
behavior is unchanged.

The dedicated source branch is `aaron/ngram-code-edit-benchmark`, based on
`aaron/dynamic-ngram-global-tree` at `c91f64717`. It includes the runner, TRT
preparation helper, and the raw-token corpus API fix required for matched inputs.

This is the Baseten-inspired reconstruction, not an exact reproduction of the
unpublished settings behind TRT PR #11434. The runner records performance and
output agreement separately; speedups alone do not establish equivalent outputs.

## Use the prebuilt SGLang environment (recommended)

Run from a standalone clone of `aaron/ngram-code-edit-benchmark` on the Linux
GPU host, outside the TRT container. **No Docker build is required.** The launcher
uses `docker.io/lmsysorg/sglang:v0.5.19`, mounts this checkout read-only, and puts
its `python/` directory first on `PYTHONPATH`. This runs our experimental source
using the release image's installed dependencies; it does not run stock NGRAM.

The release's core pins match this branch: Torch 2.13.0, FlashInfer 0.6.18,
sglang-kernel 0.4.6.post1, TVM FFI 0.1.11, Transformers 5.12.1, and Tokenizers
0.22.2. See the [release dependency file](https://github.com/sgl-project/sglang/blob/v0.5.19/python/pyproject.toml).
`check_env.py` verifies these installed versions, compiler tools, source import
path, CUDA 13 PyTorch, and visibility of exactly one GPU before each launch.
The published image still needs an initial download and a compatible host
NVIDIA driver/container integration. Environment checks do not establish
output equivalence; inspect the completed run's correctness reports too.

First export `trt-inputs.json` in the existing TRT environment as described below.
Then, on the host, set your assigned GPU UUID (obtain it with
`nvidia-smi --query-gpu=index,uuid,name --format=csv`; do not assume GPU index 0):

```bash
export BENCH_GPU_UUID=GPU-YOUR-ASSIGNED-UUID
export BENCH_INPUTS_DIR="$HOME/sa-bench"

bash benchmark/ngram/run_container.sh \
  python benchmark/ngram/bench_code_edits.py run 10 1 /work/sglang-smoke \
  --inputs-dir /work --tokens /work/trt-inputs.json --fail-on-mismatch

bash benchmark/ngram/run_container.sh \
  python benchmark/ngram/bench_code_edits.py run 100 3 /work/sglang-results \
  --inputs-dir /work --tokens /work/trt-inputs.json --fail-on-mismatch
```

Run the full matrix after inspecting the smoke results. Both output directories
must be new. Host `BENCH_INPUTS_DIR` is mounted read-write at `/work`; source is
mounted at `/workspace/sglang`. The launcher uses Podman when installed, otherwise
Docker; `CONTAINER_ENGINE=docker` or `CONTAINER_ENGINE=podman` selects explicitly.
It also recognizes a `docker` executable backed by Podman. Podman uses the
assigned UUID through NVIDIA CDI; Docker uses `--gpus device=UUID`. With no command,
`bash benchmark/ngram/run_container.sh` opens a shell after the same checks.
Keep the host checkout unchanged while a benchmark runs. Stop the TRT workload
before using that same GPU for SGLang.

The Python HTTP/text benchmark does not need the gateway or a Rust build.
Runtime Rust builds are disabled. NGRAM's modified C++ retrieval extension and
GPU kernels can still compile on first server startup, outside measured requests.
This workflow avoids rebuilding CUDA dependencies, the gateway, and the complete
SGLang image. It does not change the workload, scheduler settings, or timing rules.

The runner records the image reference, checkout commit/diff, and key package
versions. A release tag is not an immutable digest. For a fixed environment,
record the pulled image's `RepoDigests` with `podman image inspect` (or Docker),
then set `SGLANG_BENCHMARK_IMAGE=docker.io/lmsysorg/sglang@sha256:...` for subsequent
runs. Use the actual digest from that image, not an arbitrary newer release.

### Optional benchmark Dockerfile

If you want a locally named image, this thin Dockerfile reuses the same release
and checks its dependencies. It installs no packages and compiles no Rust:

```bash
docker build -f benchmark/ngram/Dockerfile -t sglang-code-edit:local .
SGLANG_BENCHMARK_IMAGE=localhost/sglang-code-edit:local \
  bash benchmark/ngram/run_container.sh
```

For a Docker daemon use `SGLANG_BENCHMARK_IMAGE=sglang-code-edit:local`; Podman
qualifies locally built image names with `localhost/`. Source and data are still
mounted by the launcher. The optional `SGLANG_BASE_IMAGE` build argument accepts
the release image's immutable digest. The root `docker/Dockerfile` is the full
source-build route; it is unnecessary for this benchmark setup.

## Run

Use the existing TRT environment to export the exact benchmark input tokens.
This calls its installed `initialize_tokenizer` and `create_dataset_from_stream`
without loading a model for inference. The example assumes the checkout is
mounted at `/sglang` and the prepared inputs at `/work`:

```bash
python /sglang/benchmark/ngram/bench_code_edits.py export-trt-inputs \
  --inputs-dir /work --output /work/trt-inputs.json
```

Then use a Linux/CUDA SGLang environment with this checkout's dependencies and
the **same model files and assigned H200**, running the frameworks sequentially.
The runner launches this checkout's Python source with the current interpreter.
Keep both runner Python files together when mounting/copying them. From the
SGLang checkout:

```bash
python benchmark/ngram/bench_code_edits.py run 10 1 /work/sglang-smoke \
  --inputs-dir /work --tokens /work/trt-inputs.json

python benchmark/ngram/bench_code_edits.py run 100 3 /work/sglang-results \
  --inputs-dir /work --tokens /work/trt-inputs.json
```

Each result directory must be new. `--model-path` overrides `/work/model` when
the same snapshot has a different mount path. The runner verifies the dataset
hash, IDs/order, model/tokenizer metadata hashes, and every prompt's local
tokenization against the TRT export, including BOS. It sends exported IDs
directly and verifies the server returns identical `prompt_token_ids`.
Model weight files are reused, not independently rehashed; retain the pinned
download and manifest together. The export records the installed TRT version
and loader hash, so exporting under a different runtime is visible.

Ten warmup requests per mode/repetition are excluded from measured results.
Warmup cycles over the selected prompts deterministically, with the same
isolation as measured requests. TRT rc25's benchmark instead samples its warmup
requests randomly and also performs internal engine warmup. Each repetition
rotates mode order; concurrency remains one. The full experiment is 100 prompts
repeated three times in each mode, not 300 distinct prompts.

The optional `--trt-results /work/smoke` reads the existing TRT report format and
adds a separately labeled table of TRT throughput/TPOT and speedup against its
own baseline. Request count and output length must match. Historical reports
do not contain enough provenance to automatically prove identical inputs,
weights, GPU, or scheduler settings. No cross-framework absolute speed ratio is
computed. Existing TRT output JSON is decoded text without token IDs; it cannot
provide a token-level cross-framework correctness comparison.
TRT also disables detokenization inside its measured request window; the
SGLang HTTP response path includes it.

## Comparison contract

| Setting | SGLang runner | Existing TRT helper |
| --- | --- | --- |
| Workload | Same prepared rows/order and exported token IDs | Pinned GlaiveAI code-edit prompts |
| Model | Same Llama-3.1-8B base snapshot, BF16 model and KV | Llama-3.1-8B base, BF16 |
| Load | TP=1, one running request | TP=1, batch/concurrency=1 |
| Sampling | Greedy, 256 tokens, ignore EOS | Same |
| Draft capacity | Width 5: current token + 4 proposals | 4 proposals |
| KV reuse | Disabled; explicitly flushed each request | Block reuse disabled |
| Graphs | Decode graph batch 1, prefill graphs disabled | Graph batch 1 |
| Attention | FlashInfer, page size 64 | TRTLLM backend in recorded smoke |
| Retrieval | Static current-prompt SAM + online Trie | Growing request-local SA/NGram history |
| Timing | Persistent local streaming HTTP | Internal benchmark request window |

Default modes:

| Mode | Allocation | Candidate ranking | Prompt SAM |
| --- | --- | --- | --- |
| `baseline` | Ordinary decoding; NGRAM omitted | — | No |
| `legacy_bfs` | Two SAM and two Trie proposal slots | Legacy BFS | Yes |
| `legacy_prob` | Two SAM and two Trie proposal slots | Legacy PROB | Yes |
| `path_probability` | Global allocation of four slots | Path probability | Yes |
| `specificity_path_probability` | Global allocation of four slots | Specificity × path probability | Yes |

Add `trie_only` with `--modes` for the optional legacy BFS control without a
prompt SAM. Any selected matrix must include `baseline`. The legacy PROB row
helps distinguish allocation changes from the BFS-versus-frequency ranking
change, although global selection still uses a different anchor policy.

Match limits are explicit: `--max-trie-depth 18`, breadth 1–4, and Trie capacity
10 million nodes. SGLang has no independent equivalent of TRT's maximum match
size 3: Trie depth also limits stored history and rollout, and SAM matching is
bounded by the query suffix. A separately labeled `--max-trie-depth 3` ablation
is available, but it is **not** algorithmic parity with TRT SA3. Width matching
also does not equate tree/chain topology, occupied proposal nodes, or GPU work.

CUDA overlap stays enabled by default for all SGLang rows, with resolved values
checked at startup. TRT's recorded NGram smoke forced overlap off while baseline
and SA kept it on. Run this separate controlled SGLang matrix:

```bash
python benchmark/ngram/bench_code_edits.py run 100 3 /work/sglang-no-overlap \
  --inputs-dir /work --tokens /work/trt-inputs.json \
  --disable-overlap-schedule
```

For its TRT counterpart, copy the prepared directory's `configs/` and set
`disable_overlap_scheduler: true` in **all four** YAMLs before invoking the
existing TRT helper in a new result directory. Preserve the original YAMLs and
resolved logs with each experiment. `--disable-cuda-graph` allows a separate
SGLang graph ablation; change the corresponding TRT graph configuration and
verify its resolved behavior before comparing. Do not mix those trials into
the default matrix.

## Isolation and measurements

```text
assert no corpus -> flush KV/Trie -> load exact current prompt IDs
                 -> verify loaded count -> stream generation
                 -> remove corpus -> assert no corpus
```

Only the server started by the runner is managed; an occupied port is rejected.
Warmup uses this lifecycle too. Flush preserves external SAMs, so removal and
registry checks are necessary. Failed loads/streams trigger removal; failed
flushes or corpus checks stop the experiment. Baseline omits the NGRAM-only
corpus endpoints. Servers are shut down between modes, including on interruption.

The token-input path required a small runtime fix: the existing HTTP schema
already exposed `token_chunks`, but the tokenizer control layer accepted only
documents/files. It now validates raw token chunks, vocabulary bounds, and the
token budget, then forwards them unchanged. Chunks are contiguous pieces of
one token stream; no implicit document separators or special tokens are added.

Every request saves text, output IDs, prompt IDs, completion count, final raw
acceptance counters, stream-event arrival times, TTFT, TPOT, HTTP latency, flush,
corpus load, setup, cleanup, and workflow time. Corpus load includes serialization,
upload, build, and acknowledgement; it does not separately instrument CPU build.
`/flush_cache` also empties the GPU allocator cache, unlike TRT's request-local
retrieval initialization, so even generation-only measurements can have different
allocator-cache conditions.

```text
generation tok/s = total output tokens / sum(HTTP generation durations)
workflow tok/s   = total output tokens / sum(setup + generation + cleanup)
window tok/s     = total output tokens / measured trial wall time
TPOT            = (last token event time - first token event time) / (N - 1)
accept length   = sum(completion_tokens) / sum(spec_verify_ct)
accept rate     = sum(spec_num_correct_drafts) / sum(spec_num_proposed_drafts)
```

Window time additionally includes client validation and artifact writes between
requests. Startup, warmup, and offline token verification have separate timers.
Do not call generation throughput total workflow throughput. Proposed-draft
counters count configured capacity, including padding; acceptance is aggregated
from final request counters rather than periodic server totals. TRT's
`decode_iteration + 1` convention and SGLang's verify count have different
first/final-step details. Multi-token events make TPOT an average, not individual
token arrival latency.

`summary.json` retains each trial, median/min/max/standard deviation of run-level
metrics, and per-repetition speedups over SGLang baseline. It compares each
speculative output to that repetition's baseline by task ID, recording mismatch
counts and the first differing token. Baseline repetitions and each mode's own
repeatability are checked separately. `--fail-on-mismatch` exits unsuccessfully
after saving the complete report if any baseline comparison differs. A one-trial
baseline self-comparison provides no repeatability evidence.

The old TRT smoke has two unresolved baseline text mismatches. Do not treat
its speedups, or a successful SGLang benchmark exit without strict comparison,
as proof of equivalent outputs. Fixed-length output with EOS ignored does not
measure edit correctness or natural stopping. The initial experiment exercises
one external source; distractors/multiple SAMs require a separate workload.

## Inspect output mismatches without rerunning inference

From the host, inspect a completed result directory using Python's standard
library. This reads the saved outputs and verifies that the summary agrees
with them; it does not start a server or load model weights:

```bash
python3 benchmark/ngram/inspect_code_edit_mismatches.py \
  "$BENCH_INPUTS_DIR/sglang-results-$BENCH_RUN_ID"
```

For decoded text around each first differing token, use the existing container
and local tokenizer. Its normal environment preflight still runs, but the
inspection itself performs no model inference:

```bash
bash benchmark/ngram/run_container.sh \
  python benchmark/ngram/inspect_code_edit_mismatches.py \
  "/work/sglang-results-$BENCH_RUN_ID" \
  --model-path /work/model \
  --output "/work/sglang-results-$BENCH_RUN_ID/mismatch-details.json"
```

The report contains unique affected task IDs, counts per repetition, same-mode
repeatability, pairwise comparisons between modes, first differing token IDs,
nearby token windows, and the original saved text. With `--model-path`, it also
includes decoded windows and tokenizer pieces. Tokenizer metadata must match
the saved input export. Special tokens are retained. Token IDs are authoritative:
decoding a window can split a multi-token UTF-8 character.

Use `--mode path_probability --task-id 17 --limit 3 --context-tokens 16` to
focus the printed examples on an actual affected task ID. These filters do not
remove data from the JSON report. Omit `--output` when repeating inspection,
or choose a new filename; existing reports are never overwritten.

For 100 requests and three repetitions, 27 mismatches means 27 of 300
request/repetition pairs, not necessarily 27 distinct prompts. Counts divisible
by three do not establish that the same prompts differ every time. Equal counts
in two modes do not establish that their actual outputs agree.

Greedy speculative decoding should emit the same tokens if the target model
produces the same predictions for the same prefixes. NGRAM verifies against
the target logits computed in a tree-shaped forward pass, whereas baseline
uses ordinary decode passes. Numerical differences across execution shapes
can change a near-tied argmax; a mask, position, cache, or acceptance bug can
also cause divergence. Existing artifacts do not contain target logits or
verification trees. Diagnosing the cause requires an instrumented replay at
the first divergence, comparing the common prefix, scores, tree ancestry,
positions, and KV state. Do not infer a harmless numerical explanation solely
from a low mismatch rate or from all speculative modes agreeing.

## Local validation

```bash
python -m unittest discover -s benchmark/ngram -p 'test_*.py' -v
python -m unittest discover -s test/registered/unit/managers \
  -p 'test_external_corpus_token_chunks.py' -v
```

The first suite uses a loopback HTTP server, without SGLang/GPU imports. The
second exercises the actual tokenizer control method in an installed SGLang
development environment. GPU startup, native cache clearing, graph execution,
and real output equivalence require the `10 x 1` smoke run.

Pinned TRT references: [dataset loader](https://github.com/NVIDIA/TensorRT-LLM/blob/785c948197b55267260fec3f7f52e47000888d0a/tensorrt_llm/bench/utils/data.py),
[throughput runner](https://github.com/NVIDIA/TensorRT-LLM/blob/785c948197b55267260fec3f7f52e47000888d0a/tensorrt_llm/bench/benchmark/throughput.py),
and [warmup helper](https://github.com/NVIDIA/TensorRT-LLM/blob/785c948197b55267260fec3f7f52e47000888d0a/tensorrt_llm/bench/benchmark/utils/general.py).
