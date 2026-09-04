# SGLang CI failure monitoring

Scripts used by [.github/workflows/ci-failure-monitor.yml](../../.github/workflows/ci-failure-monitor.yml) (scheduled failure analysis) and [.github/workflows/ci-lark-notify.yml](../../.github/workflows/ci-lark-notify.yml) (Lark notifications).

## Tools

1. **Failures Analyzer** (`ci_failures_analysis.py`): Tracks consecutive failures, identifies flaky jobs, and monitors runner health across PR Test / Nightly workflows (Nvidia, AMD, Intel, XPU, NPU).
2. **Lark Notifier** (`lark_notify.py`): Posts CUDA CI health cards to a Lark group through an incoming webhook (`LARK_WEBHOOK` secret). Stdlib only. Three subcommands:
   - `ci-status --run-id N`: one card per finished scheduled run of the Nvidia nightly / weekly / scheduled pr-test. The first attempt lists its failed jobs; a rerun (attempt N > 1) is compared with attempt N-1 of the same run (fixed by rerun / still failing). Triggered by `workflow_run`.
   - `runner-health --state-file F`: per-pool online / offline counts for the primary CUDA labels (`N-gpu-h100|h200|h20|5090|b200|b300|gb200|gb300|a10`). Posts only on degraded / recovered transitions plus an hourly reminder while degraded; state is carried between runs via `actions/cache`. Needs an admin PAT to list runners.
   - `queue-digest --hours 8 --only-if-slow`: per-pool queue time p50 / p90 / max over the window plus currently queued jobs, posted only when some pool's p90 exceeds `--slow-minutes` (default 30). Links to the latest Runner Utilization Report run.

   All subcommands accept `--dry-run` to print the card JSON instead of posting.

## Installation

```bash
pip install requests
```

## Usage

### Failures Analyzer

```bash
export GITHUB_TOKEN="your_token_here"

python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 50 --threshold 2
python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 300 --threshold 2
python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 500 --threshold 3
```

## Token permissions

The GitHub token needs `repo` and `workflow` scopes to read CI run data; otherwise API calls may return 404.

### Failures Analyzer parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--token` | Required | GitHub Personal Access Token |
| `--limit` | 500 | Number of workflow runs to analyze |
| `--threshold` | 3 | Alert threshold for consecutive failures |
| `--output` | None | Output JSON file (optional) |

## Historical note

The former **CI Monitor** workflow (`ci-monitor.yml`) and its analyzers (`ci_analyzer.py`, `ci_analyzer_perf.py`, `ci_analyzer_balance.py`) were removed as redundant; use this failure monitor workflow and scripts for ongoing CI health alerts.
