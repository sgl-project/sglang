# SGLang CI failure monitoring

Scripts used by [.github/workflows/ci-failure-monitor.yml](../../.github/workflows/ci-failure-monitor.yml): scheduled failure analysis.

## Tools

1. **Failures Analyzer** (`ci_failures_analysis.py`): Tracks consecutive failures, identifies flaky jobs, and monitors runner health across PR Test / Nightly workflows (Nvidia, AMD, Intel, XPU, NPU).

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
