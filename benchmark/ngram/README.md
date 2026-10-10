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
unpublished settings behind TRT PR #11434. The runner implements the comparison;
GPU performance and output agreement still need to be measured.

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
