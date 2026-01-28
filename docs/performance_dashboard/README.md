# SGLang Performance Dashboard

A web-based dashboard for visualizing SGLang nightly test performance metrics.

## Features

- **Performance Trends**: View throughput, latency, and TTFT trends over time
- **Model Comparison**: Compare performance across different models and configurations
- **Filtering**: Filter by GPU configuration, model, variant, and batch size
- **Interactive Charts**: Zoom, pan, and hover for detailed metrics
- **Run History**: View recent benchmark runs with links to GitHub Actions

## Quick Start

### Option 1: Run with Local Server (Recommended)

For live data from GitHub Actions artifacts:

```bash
# Install requirements
pip install requests

# Run the server
python server.py --fetch-on-start

# Visit http://localhost:8000
```

The server provides:
- Automatic fetching of metrics from GitHub
- Caching to reduce API calls
- `/api/metrics` endpoint for the frontend

### Option 2: Fetch Data Manually

Use the fetch script to download metrics data:

```bash
# Fetch last 30 days of metrics
python fetch_metrics.py --output metrics_data.json

# Fetch a specific run
python fetch_metrics.py --run-id 21338741812 --output single_run.json

# Fetch only scheduled (nightly) runs
python fetch_metrics.py --scheduled-only --days 7
```

## GitHub Token

To download artifacts from GitHub, you need authentication:

1. **Using `gh` CLI** (recommended):
   ```bash
   gh auth login
   ```

2. **Using environment variable**:
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

Without a token, the dashboard will show run metadata but not detailed benchmark results.

## Data Structure

The metrics JSON has this structure:

```json
{
  "run_id": "21338741812",
  "run_date": "2026-01-25T22:24:02.090218+00:00",
  "commit_sha": "5cdb391...",
  "branch": "main",
  "results": [
    {
      "gpu_config": "8-gpu-h200",
      "partition": 0,
      "model": "deepseek-ai/DeepSeek-V3.1",
      "variant": "TP8+MTP",
      "benchmarks": [
        {
          "batch_size": 1,
          "input_len": 4096,
          "output_len": 512,
          "latency_ms": 2400.72,
          "input_throughput": 21408.64,
          "output_throughput": 231.74,
          "overall_throughput": 1919.43,
          "ttft_ms": 191.32,
          "acc_length": 3.19
        }
      ]
    }
  ]
}
```

## Deployment

### GitHub Pages

The dashboard can be deployed to GitHub Pages for public access:

1. Copy the dashboard files to `docs/performance_dashboard/`
2. Enable GitHub Pages in repository settings
3. Set up a GitHub Action to periodically update metrics data

### Self-Hosted

For a self-hosted deployment with live data:

1. Set up a server running `server.py`
2. Configure a cron job or systemd timer to refresh data
3. Optionally put behind nginx/caddy for SSL

## Metrics Explained

- **Overall Throughput**: Total tokens (input + output) processed per second
- **Input Throughput**: Input tokens processed per second (prefill speed)
- **Output Throughput**: Output tokens generated per second (decode speed)
- **Latency**: End-to-end time to complete the request
- **TTFT**: Time to First Token - time until the first output token
- **Acc Length**: Acceptance length for speculative decoding (MTP variants)

## Contributing

To add support for new metrics or visualizations:

1. Update `fetch_metrics.py` if data collection needs changes
2. Modify `app.js` to add new chart types or filters
3. Update `index.html` for UI changes

## Troubleshooting

**No data displayed**
- Check browser console for errors
- Verify GitHub API is accessible
- Try running with `server.py --fetch-on-start`

**API rate limits**
- Use a GitHub token for higher limits
- The server caches data for 5 minutes

**Charts not rendering**
- Ensure Chart.js is loading from CDN
- Check for JavaScript errors in console
