# SGLang CI failure monitoring

Scripts used by [.github/workflows/ci-failure-monitor.yml](../../.github/workflows/ci-failure-monitor.yml): scheduled failure analysis and optional Slack notifications.

## Tools

1. **Failures Analyzer** (`ci_failures_analysis.py`): Tracks consecutive failures, identifies flaky jobs, and monitors runner health across PR Test / Nightly workflows (Nvidia, AMD, Intel, XPU, NPU).

2. **Slack poster** (`post_ci_failures_to_slack.py`): Sends a condensed summary from a failure-analysis JSON to Slack (invoked by the workflow when `SGLANG_DIFFUSION_SLACK_TOKEN` is set).

## Installation

```bash
pip install requests slack_sdk
```

(`slack_sdk` is only required for `post_ci_failures_to_slack.py`.)

## Usage

### Failures Analyzer

```bash
export GITHUB_TOKEN="your_token_here"

python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 50 --threshold 2
python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 300 --threshold 2
python ci_failures_analysis.py --token $GITHUB_TOKEN --limit 500 --threshold 3
```

### Slack notifications

From the `scripts/ci_monitor` directory, after generating a report:

```bash
export SGLANG_DIFFUSION_SLACK_TOKEN="xoxb-..."
python post_ci_failures_to_slack.py --report-file ci_failure_analysis_YYYYMMDD_HHMMSS.json
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
