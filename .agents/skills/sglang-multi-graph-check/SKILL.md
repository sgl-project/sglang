---
name: sglang-multi-graph-check
description: Investigate SGLang CUDA-graph lifetime and replay correctness for a serving configuration, including multiple graphs or variants sharing buffers, or inspect and re-judge archived graph-check runs. Use for explicit graph-lifetime or multi-graph investigations, not ordinary code reviews, routine benchmarks or general serving debugging.
---

# SGLang multi-graph check

Use `python -m sglang.test.cuda_graph_lifetime`, the single in-tree implementation.
Read [the CLI reference](references/cli.md) before invoking it or interpreting an archive.
Report lifetime, numerical and coverage verdicts separately; never infer one from another.

## Which of three jobs is this

1. **New execution:** use the workflow below with a cached model, idle GPUs and an agreed budget.
2. **Read-only archive inspection:** read the saved evidence without changing it; skip the
   entire new-execution workflow. No GPU or serving checkout is required to report what it says.
3. **Archive recomputation:** `--recompute` writes a new result and renames any previous
   recomputed result, as described in the reference. If receipts must remain completely
   unchanged, inspect read-only or use an authorized working copy. Skip the new-execution workflow.

## Workflow for a new execution

1. **Record identities:** the serving checkout's commit/tree and dirty status, Python, torch
   and sglang versions. Distinguish the harness checkout from the source each worker imports.
2. **Agree on scope:** the serving command, model/adapters, backends, parallelism, graph knobs,
   traffic and required variants. Pass launch knobs through `--env`. Budget three launches
   (eager, ordinary, instrumented), or four with `--repeat-reference`; explicit skip flags can
   reduce this. Do not run unconfirmed configurations or download missing models.
3. **Run the execution-path preflight** below; settle labels, traffic and material gaps before
   launching. Source inspection predicts coverage, not proof that a path ran.
4. **Check device availability without disturbing anyone.**
   `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and
   `--query-compute-apps=pid,process_name --format=csv` show which GPUs are free; choose those
   with `CUDA_VISIBLE_DEVICES` (also passed with `--env`). Never stop another process to make
   room, and never acquire or release machines.
5. **Invoke** from the serving checkout with `PYTHONPATH=<checkout>/python` and a fresh `--out`.
6. **Reconcile and report:** check resolved dispatch, worker identities, inspection findings and
   per-worker replay labels against the preflight. Update the path map with unverified paths
   and unreplayed buckets; do not upgrade a verdict based on source review. Read the ledger,
   not just the exit code; the reference identifies its fields and evidence limitations.
7. **Preserve receipts:** retain the run directory. Export logs and reports only within the
   authorized scope; do not export the compared `*.outputs.json` numerical arrays without agreement.

## Execution-path preflight (before a new serving run)

Follow only this configuration's reachable paths, not the entire repository or every layer's
arithmetic. Use the current source and [reference](references/cli.md#where-coverage-comes-from):

- Resolve command/config/env, cached model configuration and hardware-dependent defaults.
  Mark automatic dispatch unresolved until confirmed by the server's resolved arguments.
- Trace worker topology, model runner, decode/prefill capture, attention and LoRA/MoE backends,
  enabled draft/verify paths, communication/workspace ownership, and nested or independent graphs.
- Check injection, capture/replay interception and allocator visibility. Classify each path as
  lifetime-inspectable, replay-observable only, partial, unsupported or unresolved. A component
  inside an inspected parent graph needs no separate hook; an injection marker alone proves no
  inspection. Cite source locations for non-obvious conclusions.
- Derive existing labels and traffic from the map, within budget. Require variant labels through
  `--expect-labels`; verify exact buckets from raw per-worker replay labels (sizes are normalized
  out of requirements). No blanket two-graph requirement, invented labels or silent matrix expansion.

| Selected component/path | Capture/replay mechanism | Injection/audit evidence | Relevant limits or uncertainty |
|---|---|---|---|

Report material gaps before launching; partial diagnostics must stay partial. After execution,
check the map against resolved dispatch, captures, inspections and actual replays on every worker.
For archive jobs, use recorded arguments, topology, coverage declaration and worker source plus
matching source where available; mark absent information unknown and skip all launch/device steps.

## Report template

```
Job: <new execution | read-only inspection | recomputation (in place | in a working copy)>
Configuration: <serving command, env knobs, GPUs, checkout commit/tree, dirty or clean, tool digest>
Run: <run id and directory | the archived run and which file was read>
lifetime:  <verdict | NOT-RUN | NOT-RECOMPUTED> (<identity>) — <reason or "clean: N captures, N replays, N inspections per worker">
numerical: <verdict | NOT-RUN>, kind <kind>, evidence <verified | unverified-legacy> — <reason>; control <applied | unusable | not run>
coverage:  <verdict> (<identity>, requirements <provenance>) — required <list>; replayed <requested graphs>; replay unobserved <captured_not_replayed>
agreement: instrumented vs ordinary <newly recomputed | historical only (not re-verified) | rejected | unavailable>
Provenance: <source identity statement from the ledger>
Execution path: <prediction versus worker reports; dispatch differences; paths/buckets unexecuted or unverified>
Path map: <the table, or where it is kept>
Not run / not covered: <skipped launches and what they leave out, limitations that apply>
```

## Boundaries, stated in every report

- A PASS holds only for the recorded configuration and the traffic the sweep sent.
- INCONCLUSIVE, INCOMPLETE, NOT-RUN and NOT-RECOMPUTED are not PASS, and never become one.
- The batch-sensitivity control is a diagnostic heuristic (eager under another batch
  composition), not proof of numerical correctness.
- The lifetime guarantee does not reach device-side pointer tables, memory outside the PyTorch
  caching allocator, metadata that stays allocated but is no longer refreshed, or traffic the
  sweep did not send.
- A successful process exit is not evidence that every verdict passed; only the ledger is, and
  `--recompute` exits 0 whatever its verdicts.
- This skill authorizes no dependency change, model download, node acquisition or release,
  publishing, CI dispatch, or expansion of the test matrix beyond the confirmed configuration.
