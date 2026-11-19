---
name: clean-startup-log
description: Audit SGLang startup logs, save evidence, and propose cleanup for user review. With no arguments, run Qwen3-8B at TP1 and TP2 plus gpt-oss-20b at TP1.
disable-model-invocation: true
---

# Audit SGLang Startup Logs

The default outcome is saved logs and a findings report. Apply runtime changes
only after the user selects them. A request to edit this skill does not itself
launch servers.

## Default runs

A bare `$clean-startup-log` invocation runs these cases sequentially without
asking for commands. Explicit commands, models, or TP sizes replace this matrix.

| Case / log filename | Command |
|---|---|
| `qwen3-8b-tp1.log` | `uv run sglang serve --model-path Qwen/Qwen3-8B` |
| `qwen3-8b-tp2.log` | `uv run sglang serve --model-path Qwen/Qwen3-8B --tp 2` |
| `gpt-oss-20b-tp1.log` | `uv run sglang serve --model-path openai/gpt-oss-20b` |

These cover dense, tensor-parallel, and MoE/hybrid sliding-window attention
startup. Reuse complete captures from the current audit when code and environment
have not changed.

## Capture logs

1. Check the checkout, free GPUs, and ports once. Use the requested command when
   resources are free; otherwise select free GPUs with `CUDA_VISIBLE_DEVICES` and
   an unused `--port`, recording the adjustments. Leave existing servers alone.
2. Create a unique directory with `mktemp -d /tmp/sglang-startup-audit-XXXXXX`.
   Save raw stdout and stderr together in a separate log for each case, for
   example with `set -o pipefail` and `COMMAND 2>&1 | tee LOG_PATH`. Record commands,
   GPU IDs, ports, commit, relevant overrides, and readiness status.
3. Wait for `The server is fired up and ready to roll!`, then stop that server and
   its workers before the next case. First runs can spend many minutes downloading
   weights or compiling FlashInfer kernels; check download/compiler activity
   before treating a quiet log as a hang. Preserve partial logs for failed or
   stalled starts and report the last stage. Continue independent cases when possible.
4. Preserve the user's logging configuration, including `NCCL_DEBUG`. If NCCL
   verbosity needs explaining, inspect relevant shell settings, `NCCL_CONF_FILE`,
   and `/etc/nccl.conf`. Do not override intentional diagnostics or recommend
   `NCCL_DEBUG=WARN` solely because the output is long. Avoid full environment dumps.

## Investigate efficiently

- Scan for deprecations, duplicate handler output, unrelated import failures,
  unformatted prints, and unexpected warnings. Read representative excerpts and
  counts instead of repeatedly dumping `server_args`, progress redraws, or NCCL
  diagnostics. Normalize carriage returns for analysis only; preserve raw logs.
- Trace each candidate to its actual emitter with focused `rg` searches. Inspect
  its log level: SGLang's formatter may omit severity. Group shared signatures
  across cases and distinguish handler duplication from separate GPU/process calls.
- Repetition, WARNING severity, or a different third-party format alone does not
  establish a cleanup need. Consider whether the message explains configuration,
  progress, resource use, or an operational limitation.
- Consult [noise-source hints](references/noise-sources.md) only for a matching
  signature or an unresolved emitter. Verify current code rather than trusting
  historical line numbers, fix status, or assumptions about unrelated models.

## Accepted output

Preserve these reviewed messages unless the user requests a different policy:

- NCCL diagnostics enabled by the user's environment or host configuration.
- NUMA permission warnings, including one check per GPU in TP runs.
- GPT-OSS MXFP4 backend-selection warnings and default page-size selection warnings.
- `Init Unified Radix Cache. Components: ... Tree Core: ...`, tree-cache summaries,
  SWA allocation details, and per-rank memory/timing records.
- Useful progress bars, warmup HTTP access logs, uv synchronization messages,
  isolated NCCL/Gloo startup lines, and one timestamped HF authentication warning.

These can appear in a clean startup log. Do not repeatedly propose the declined
NUMA deduplication, backend/page-size level changes, or NCCL verbosity override.
Keep real operational warnings visible: for example, a Harmony vocabulary failure
can disable `/v1/responses` even when server readiness and `/generate` succeed.

## Report before changing code

Return a compact run table with readiness status and clickable raw-log links.
For each actual cleanup candidate, give an exact representative message, affected
cases/counts, source file/function, and specific proposed behavior. Distinguish
confirmed findings from suspicions and operational failures from logging noise.

If there are no actionable cleanup findings, say the logs are clean and no
further cleanup is needed. Otherwise, ask which numbered changes to adopt and
wait for the user's selections before editing runtime code or preparing patches.
Honor existing approvals and declined items without asking again.

## Apply selected changes

- Batch compatible approved edits, then verify affected cases once. Repeat a
  startup only for a new change, failure, or unresolved concern; do not relaunch
  after every one-line edit. Save verification logs separately from baselines.
- Preserve useful warnings and application handlers. HF can warn during early
  CLI model detection before `configure_logger()`, and spawned processes have
  independent logger state. Keep `configure_hf_hub_logger()` in both
  `suppress_noisy_warnings()` and `configure_logger()`; make repeated setup safe.
- Keep the legacy compiled-kernel cache migration notice at DEBUG. Avoid broad
  library-level suppression or fd redirection for a narrow logging problem.
- Run relevant formatting and focused existing checks. Add tests only when they
  verify meaningful behavior, not a log-level spelling. Report changes and
  verification; create branches, commits, and PRs when requested.
