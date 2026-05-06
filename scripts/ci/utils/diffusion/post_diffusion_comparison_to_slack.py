#!/usr/bin/env python3
"""Post diffusion cross-framework comparison summary to Slack.

Reads ``comparison-results.json`` produced by ``run_comparison.py`` and posts a
compact summary message to a Slack channel.
"""

import argparse
import importlib
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diffusion CI channel.
SLACK_CHANNEL_ID = "C0A02NDF7UY"

# Published dashboard location in sglang-ci-data.
DASHBOARD_URL = (
    "https://github.com/sglang-bot/sglang-ci-data/blob/main/"
    "diffusion-comparisons/dashboard.md"
)


def _short_sha(sha: str) -> str:
    if not sha or sha == "unknown":
        return "unknown"
    return sha[:7]


def _build_case_rows(results: list[dict]) -> list[str]:
    """Build per-case summary rows for Slack."""
    by_case: dict[str, dict[str, float | None]] = {}
    case_errors: dict[str, dict[str, str]] = {}
    for item in results:
        cid = item.get("case_id", "unknown-case")
        fw = item.get("framework", "unknown-fw")
        by_case.setdefault(cid, {})[fw] = item.get("latency_s")
        if item.get("error"):
            case_errors.setdefault(cid, {})[fw] = str(item["error"])

    rows: list[str] = []
    for cid in sorted(by_case.keys()):
        lat_map = by_case[cid]
        err_map = case_errors.get(cid, {})
        valid = {fw: lat for fw, lat in lat_map.items() if lat is not None}

        if not valid:
            details = []
            for fw in sorted(lat_map.keys()):
                err = err_map.get(fw, "N/A")
                details.append(f"{fw}=ERR({err[:60]})")
            rows.append(f"- `{cid}`: {' | '.join(details)}")
            continue

        fastest_fw, fastest_lat = min(valid.items(), key=lambda kv: kv[1])
        parts = [f"{fw}={lat:.2f}s" for fw, lat in sorted(valid.items())]
        suffix = ""
        sglang = valid.get("sglang")
        if sglang is not None:
            slower = [
                fw for fw, lat in valid.items() if fw != "sglang" and sglang > lat
            ]
            if slower:
                suffix = f" | :warning: slower than {', '.join(sorted(slower))}"
        rows.append(
            f"- `{cid}`: fastest `{fastest_fw}` {fastest_lat:.2f}s | "
            f"{' | '.join(parts)}{suffix}"
        )
    return rows


def post_diffusion_comparison_to_slack(results_file: str) -> bool:
    """Post comparison summary to Slack."""
    token = os.environ.get("SGLANG_DIFFUSION_SLACK_TOKEN")
    if not token:
        logger.info("Slack post skipped: no SGLANG_DIFFUSION_SLACK_TOKEN")
        return True

    WebClient = importlib.import_module("slack_sdk").WebClient

    channel = os.environ.get("SGLANG_DIFFUSION_SLACK_CHANNEL_ID", SLACK_CHANNEL_ID)

    result_missing = not os.path.exists(results_file)
    if result_missing:
        payload = {
            "timestamp": "",
            "commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "results": [],
        }
    else:
        with open(results_file) as f:
            payload = json.load(f)

    results = payload.get("results", [])
    timestamp = payload.get("timestamp", "")
    sha = payload.get("commit_sha", "unknown")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server_url}/{repo}/actions/runs/{run_id}" if run_id else ""

    rows = _build_case_rows(results)
    failed_entries = sum(1 for r in results if r.get("latency_s") is None)
    has_failure = result_missing or failed_entries > 0
    status_emoji = ":red_circle:" if has_failure else ":large_green_circle:"
    color = "danger" if has_failure else "good"

    lines = [
        f"{status_emoji} *Diffusion cross-framework comparison*",
        f"Commit: `{_short_sha(sha)}`",
        f"Timestamp: `{timestamp}`" if timestamp else "",
        f"Run: <{run_url}|GitHub Actions>" if run_url else "",
        f"Dashboard: <{DASHBOARD_URL}|latest dashboard>",
        f"Results: total `{len(results)}`, failed `{failed_entries}`",
        f":warning: Results file missing: `{results_file}`" if result_missing else "",
        "",
        "*Case Summary*",
        *(rows or ["- No comparison results found."]),
    ]
    text = "\n".join([line for line in lines if line])

    # Slack message text limit safety.
    if len(text) > 3900:
        head = "\n".join(lines[:7])
        body_lines = rows
        text = head
        client = WebClient(token=token, timeout=60)
        resp = client.chat_postMessage(
            channel=channel,
            text=text,
            attachments=[
                {
                    "color": color,
                    "footer": f"SGLang diffusion comparison | {datetime.now().isoformat(timespec='seconds')}",
                    "ts": int(datetime.now().timestamp()),
                }
            ],
        )
        thread_ts = resp.get("ts")
        if not thread_ts:
            return True
        chunk = ""
        for row in body_lines:
            if len(chunk) + len(row) + 1 > 3800:
                client.chat_postMessage(
                    channel=channel, thread_ts=thread_ts, text=chunk
                )
                chunk = row
            else:
                chunk = f"{chunk}\n{row}" if chunk else row
        if chunk:
            client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=chunk)
        return True

    client = WebClient(token=token, timeout=60)
    client.chat_postMessage(
        channel=channel,
        text=text,
        attachments=[
            {
                "color": color,
                "footer": f"SGLang diffusion comparison | {datetime.now().isoformat(timespec='seconds')}",
                "ts": int(datetime.now().timestamp()),
            }
        ],
    )
    logger.info("Diffusion comparison summary posted to Slack")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post diffusion comparison summary to Slack"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to comparison-results.json",
    )
    args = parser.parse_args()

    success = post_diffusion_comparison_to_slack(args.results_file)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
