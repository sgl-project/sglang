# Raw validation artifacts

Machine-readable summaries for `results/glm53_flash_kda_ptpc_results.md`.

- `serving_{bf16,ptpc}_c*.json`: PR #33602-style 8K/1K serving runs.
- `component_*.json`: raw event-timed BF16/PTPC samples.
- `profile_{bf16,ptpc}_tp0_extend.csv`: functional trace attribution.
- `profile_{bf16,ptpc}_triage.txt`: unified profiler summaries.
- `accuracy_{bf16,ptpc}_tp{4,8}_metrics.json`: GSM8K metrics.

Large Chrome traces and per-example GSM8K predictions remain on the validation
host; their exact paths are recorded in the main report and PR description.
