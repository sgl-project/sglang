# srtslurm Log Analysis

You are analyzing logs from a failed srtslurm job. srtslurm is a Python-first
orchestration framework for running distributed LLM inference benchmarks on
SLURM clusters using SGLang and TRTLLM backends.

## Quick Start

1. List the directory contents to understand what files are present.
2. Read files in priority order.
3. Correlate timestamps to identify the real failure point.
4. Distinguish root cause from noisy warnings.

## Priority Order

### 1. `sweep_{job_id}.log`

Read this first. It is the orchestration timeline.

Look for:
- stage transitions
- worker readiness
- benchmark start
- exit codes
- the last error before teardown

### 2. `benchmark.out`

If present, this usually contains the benchmark-side exception or timeout.

### 3. `artifacts/*/logs/aiperf_*.log`

If present, these often contain framework-level initialization failures and
HTTP/network issues.

### 4. Worker logs

Focus on errors that line up with the failure timestamp:
- `{node}_prefill_w{N}.out`
- `{node}_decode_w{N}.out`
- `{node}_frontend_{N}.out`

### 5. `infra.out`

Use this to confirm infrastructure failures involving NATS, etcd, ports, or
service health checks.

## Timestamp Correlation

This is the most important rule.

Many warnings are harmless. The root cause is usually the error that occurs at
the same time the orchestration log transitions into failure.

Use this method:
1. Find the failure time in `sweep_{job_id}.log`.
2. Search other logs for matching timestamps.
3. Ignore earlier warnings if the job continued past them.

## Common Signal

High-signal failures:
- `ReadTimeout`
- `Connection refused`
- `CUDA out of memory`
- `NCCL timeout`
- `Model not found`
- benchmark exit code failures

Low-signal noise:
- dependency resolver warnings
- cleanup warnings during teardown
- keep-alive failures after the main crash
- import warnings unrelated to the active model

## Output Format

Write markdown with this structure:

```markdown
## Job Analysis: {job_id}

### Root Cause
...

### Evidence
- `file:line or file`
- timestamp
- relevant error text

### Timeline
- key event -> timestamp

### Noise
- warnings that were not causal

### Suspect PRs (sglang)
- PR #NNNN: "<title>" — reason this could be related
(only if the failure may originate in sglang)

### Recommended Fix
...
```

Keep the report concrete. Avoid generic summaries. If you are unsure, say so
and explain what evidence is missing.

## Filing Issues

After completing your analysis, if the root cause is actionable and clearly
attributable to a specific repo, open a GitHub issue using `gh issue create`.

**Rules:**
- Only file an issue if you have concrete evidence (specific error, file, line).
  Do NOT file issues for flaky infra, transient timeouts, or unclear failures.
- One issue per root cause. Do not create duplicates — search existing issues
  first with `gh issue list --repo <repo> --search "<keywords>"`.
- File against the correct repo:
  - **`NVIDIA/srt-slurm`**: orchestration bugs, config handling, srtctl behavior,
    recipe/YAML issues, incorrect flags or environment variables being passed to
    workers, worker launch failures caused by srt-slurm itself.
  - **`sgl-project/sglang`**: Do NOT auto-file issues here. Instead, use `gh` to
    review recent commits from the past day on the sglang repo:
    ```
    gh api repos/sgl-project/sglang/commits?since=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ)&per_page=50
    ```
    Identify any commits/PRs that could plausibly have caused the failure based
    on the files changed and the error you found. List these as "Suspect PRs" in
    the report with links and a brief explanation of why each is suspicious. Let
    the human decide whether to follow up.
- Use this format for srt-slurm issues:
  ```
  gh issue create --repo NVIDIA/srt-slurm \
    --title "<concise title>" \
    --body "<body>"
  ```
- The issue body should include:
  - A short summary of the failure
  - The exact error message and which log file it came from
  - The job ID and relevant config (model, flags, etc.)
- Do NOT include API keys, tokens, or secrets in the issue.
- If you are unsure which repo to file against, or if the failure is ambiguous,
  do NOT file an issue. Just note it in the report.
