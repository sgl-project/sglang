# CUDA-graph lifetime CLI reference

Run `python -m sglang.test.cuda_graph_lifetime` from the serving checkout with
`PYTHONPATH=<checkout>/python`.

## Flags

| flag | meaning |
|---|---|
| `--serving-cmd "<cmd>"` | the exact `python -m sglang.launch_server ...` command, with `--host`/`--port` |
| `--out DIR` | parent directory; every artifact goes under `DIR/<run id>/` |
| `--env KEY=VALUE` | applied to every launch (repeatable): `SGLANG_ENABLE_LORA_DUAL_GRAPH=1`, `CUDA_VISIBLE_DEVICES=...` |
| `--expect-labels a,b` | substrings that must appear in a replayed graph label on every worker, e.g. `variant_label=lora,variant_label=nolora`; never a `size=` or `rows=` part, which is normalized out before matching and so could never be met (bucket coverage is read off the raw `label_replays` instead) |
| `--expect-workers N` | a declaration that must agree with the topology the server resolves (the count itself comes from the server) |
| `--lora-names a,b` | adapter names to exercise (default: parsed from `--lora-paths`) |
| `--startup-timeout S` | seconds to wait for each launch's `/health` (default 1800) |
| `--batch-sizes 8,4,2,1` | traffic groups; `--max-new-tokens 16`; `--check-every 10` inspections cadence |
| `--repeat-reference` | the batch-sensitivity control: eager again with every request sent alone |
| `--skip-reference` | no eager reference: numerical is NOT-RUN |
| `--skip-ordinary` | no plain launch: numerical compares the instrumented launch against the eager reference; there is no instrumented-vs-ordinary agreement |
| `--skip-instrumented` | no tracer launch: lifetime is NOT-RUN and coverage is INCOMPLETE (no worker report) |
| `--recompute DIR` | re-judge an archived run (a run directory, or an `--out` with exactly one) into `ledger.recomputed.json`; an existing one is renamed to `ledger.recomputed.<tool digest>.json` first |

Give the serving command its own free `--port`: without one every launch uses 30000, and a stale
server there would answer the health probe in the new launch's place.

Exit codes: a new execution returns 0 only when all three verdicts are PASS and 1 otherwise;
`--recompute` returns 0 whenever it completes, whatever its verdicts. Neither replaces reading
the result file.

## The ledger of a run (`<out>/<run id>/ledger.json`)

- `verdicts.lifetime`: `verdict` PASS / FAIL / NO-WORKERS / NOT-RUN, `identity` (verified,
  unverified-legacy, unsupported), `problems`, per-worker `workers[]` with ranks, device UUID,
  flush nonce, counts.
- `verdicts.numerical`: `verdict` PASS / FAIL / INCONCLUSIVE / NOT-RUN, `kind` (malformed, crash,
  drift, insufficient-evidence, batch-sensitivity, not-run), `max_drift` in nats against a 1.0
  bound, `problems`, `batch_sensitivity_control` (applied, NOT-RUN when its launch failed).
- `verdicts.coverage`: `verdict` PASS / INCOMPLETE, `required`, `per_worker[].missing`,
  `captured_not_replayed` (raw labels with no recorded label replay), `label_replays`,
  `instrumented_vs_ordinary` (a fresh comparison when both launches ran), `requirement_provenance`,
  `identity`.
- `topology`: tp/pp/dp/attention-dp sizes resolved from the server log, expected workers and
  identities, or `unsupported` with the reason (multi-node, replicas with attention-DP).
- `evidence`: `tool` digest, `outputs` digests per phase, `harness_cwd_git`, `harness_cwd_dirty`,
  `worker_source`, `source_identity`.
- `coverage_declaration`, `limitations`, `claim`.

## The result of a recomputation (`ledger.recomputed.json`)

A different structure from the ledger: `original_verdicts` (what the ledger said), `verdicts`
(the re-judgment), `outputs` (provenance per phase: `verified`, `unverified-legacy` for outputs
recorded without a digest, `mismatch`, `missing`, `never-archived`, `failed-phase`), `notes`,
`tool` (the digest of the tool that re-judged). Read it with these rules:

- `lifetime` is NOT-RECOMPUTED when the reports predate worker identity (tracer version before 3);
  the original verdict is then history, not a re-judgment.
- `numerical` adds `evidence_provenance`: `unverified-legacy` outputs are judged but never count as
  verified evidence; `kind` may also be `evidence-mismatch` or `evidence-missing` (the required
  reference or candidate outputs were rejected: INCONCLUSIVE, never a verdict), and
  `batch_sensitivity_control.verdict` may be `UNUSABLE` (the optional control was rejected; the
  verdict stands on the required outputs alone).
- `coverage.instrumented_vs_ordinary` has four states: newly recomputed from both outputs
  (`kind` drift and friends); historical only (`kind` not-recomputed, `historical: true`: the
  archive never saved the outputs, and a historical PASS is not a newly verified comparison);
  rejected (`kind` evidence-rejected: recorded outputs did not match or were missing, nothing
  inherited); unavailable (no entry: one of the two launches never ran).

## Where coverage comes from

For preflight, follow the selected runner in `srt/model_executor/runner/` and `runner_backend/`,
the selected attention/LoRA/MoE backend and any draft/verify or communication path it invokes.
Check actual variant selection in that source: automatic LoRA dual graphs use the opt-in knob,
but explicit requests such as UNO are separate. Do not infer a path from the model name alone.

- `inject/sitecustomize.py` arms the tracer in every Python process of the serving command and
  wraps the modules named in `_SGLANG_MODULES` (`tracer.py`) as they are imported: the full and
  breakable graph backends and the breakable capture context (`runner_backend/`), the dedup
  registry (`cuda_graph_dedup_mixin.py`), the row-bucket runner and the metadata-glue graph
  (`runner/`). Labels come from those wrappers: `<Runner> size=<n> [variant_label=...]
  [attention_variant=...]` for the decode and prefill runners, `RowBucketGraphRunner rows=<n>`,
  `MetadataGlueGraph`. The row-bucket and metadata-glue wrappers label captures only: replay is
  unobserved by their label mechanism, not proved absent. Their labels cannot satisfy replay
  requirements. A path without a label wrapper is still inspected if it uses
  `torch.cuda.CUDAGraph`, but its generic `graph#N` label does not reliably identify a component.
  Captures outside `torch.cuda.CUDAGraph` are invisible unless separately intercepted.
- `tracer.py` wraps `torch.cuda.CUDAGraph` (capture begin and end, replay, reset) and the dedup
  registry's direct `cudaGraphLaunch`, records every captured graph with `keep_graph=True`, and
  runs the checks at safe points (first replay, every N replays, the end of an outer capture, the
  harness's flush request).
- `classifier.py` reads the kernel nodes of each raw graph and classifies every device-memory
  operand against the allocator history: static inputs, shared-pool temporaries, eager bridges of
  a captured forward, intermediates, outlived buffers, dangling ones. It cannot see pointers kept
  inside other device buffers, memory the PyTorch caching allocator did not allocate (NCCL,
  library workspaces), or a buffer that stays allocated while a graph keeps reading stale
  contents; a graph mutated after its first inspection is not read again.
- Worker-report fields that show inspection happened: `counts.captures`, `counts.checks`,
  `findings.graphs` (graphs whose operands were read), `findings.failed` and `findings.incomplete`
  (graphs that could not be read; they fail the lifetime verdict). `label_replays` shows which
  labelled graphs ran; it says nothing about inspection by itself.

Reconcile the predicted path with resolved arguments, worker identities and inspection findings.
For a requested bucket, check each worker's raw
`phases.instrumented.worker_reports[*].label_replays`; coverage requirements normalize sizes out.
Aggregate counts alone do not establish a particular component's inspection or replay. Keep any
unresolved mapping or unsupported allocation path explicit in the final assessment.

## Reading a verdict

- lifetime FAIL lists each dangling operand with the graph label, the allocation and the free stack.
- numerical FAIL of kind drift names the request, position and token with both logprobs; of kind
  malformed or crash it is the launch, not the model; INCONCLUSIVE never becomes PASS.
- coverage INCOMPLETE names the worker and missing requirement; `captured_not_replayed` is
  informational. Distinguish an unreplayed bucket from a path whose replay has no label counter.
