#!/usr/bin/env python3
"""
Post CI auto-bisect results to Slack.

Reads the bisect_results.json produced by ci_auto_bisect.py and posts
a summary message with threaded details to the CI failures Slack channel.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CI failures channel (same as post_ci_failures_to_slack.py)
SLACK_CHANNEL_ID = "C0A2DG0R7CJ"

CLASSIFICATION_EMOJI = {
    "code_regression": "🔴",
    "hardware_issue": "🟠",
    "environment_change": "🟡",
    "flaky_test": "🔵",
    "unknown": "⚪",
}

CLASSIFICATION_LABELS = {
    "code_regression": "Code Regression",
    "hardware_issue": "Hardware Issue",
    "environment_change": "Environment Change",
    "flaky_test": "Flaky Test",
    "unknown": "Unknown",
}


def post_bisect_to_slack(report_file: str) -> bool:
    """Post bisect results to Slack with threaded details."""
    from slack_sdk import WebClient

    token = os.environ.get("SGLANG_DIFFUSION_SLACK_TOKEN")
    if not token:
        logger.info("Slack post skipped: no token")
        return False

    with open(report_file) as f:
        report = json.load(f)

    try:
        results = report.get("results", [])
        summary = report.get("summary", {})
        total_analyzed = report.get("total_failures_analyzed", 0)
        total_tokens = report.get("total_tokens_used", 0)
        error_msg = report.get("error")

        client = WebClient(token=token)
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        workflow_url = ""
        if run_id:
            workflow_url = (
                f"https://github.com/sgl-project/sglang/actions/runs/{run_id}"
            )

        # Build summary message
        if error_msg:
            mentions = "<@U09R55D8EAY> <@U09ABMCKQPM>"
            summary_text = (
                f"{mentions} 🚨 *CI Auto Bisect Failed*\n" f"Error: `{error_msg[:200]}`"
            )
            if workflow_url:
                summary_text += f"\n<{workflow_url}|View logs>"
            color = "danger"
        elif total_analyzed == 0:
            summary_text = "✅ *CI Auto Bisect*: No failures requiring analysis"
            if workflow_url:
                summary_text += f"\n<{workflow_url}|View run>"
            color = "good"
        else:
            mentions = "<@U09R55D8EAY> <@U09ABMCKQPM>"
            lines = [f"{mentions} 🔍 *CI Auto Bisect Results*"]
            lines.append(f"Analyzed {total_analyzed} failures:\n")

            # Order: regressions first (most actionable)
            class_order = [
                "code_regressions",
                "hardware_issues",
                "environment_changes",
                "flaky_tests",
                "unknown",
            ]
            class_to_key = {
                "code_regressions": "code_regression",
                "hardware_issues": "hardware_issue",
                "environment_changes": "environment_change",
                "flaky_tests": "flaky_test",
                "unknown": "unknown",
            }

            for cls_key in class_order:
                count = summary.get(cls_key, 0)
                if count > 0:
                    cls = class_to_key[cls_key]
                    emoji = CLASSIFICATION_EMOJI.get(cls, "⚪")
                    label = CLASSIFICATION_LABELS.get(cls, cls)
                    lines.append(f"  {emoji} *{label}*: {count}")

            if workflow_url:
                lines.append(f"\n<{workflow_url}|View full bisect report>")

            summary_text = "\n".join(lines)

            # Color based on most severe classification
            if summary.get("code_regressions", 0) > 0:
                color = "danger"
            elif (
                summary.get("hardware_issues", 0) > 0
                or summary.get("environment_changes", 0) > 0
            ):
                color = "warning"
            else:
                color = "#439FE0"  # Blue for flaky only

        # Post parent message
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=summary_text,
            attachments=[
                {
                    "color": color,
                    "footer": f"SGLang CI Auto Bisect | {total_tokens} tokens used",
                    "footer_icon": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                    "ts": int(datetime.now().timestamp()),
                }
            ],
        )

        thread_ts = response.get("ts")
        if not thread_ts:
            logger.warning("Slack response missing 'ts', cannot post thread")
            return True

        # Post detailed breakdown in thread if there are results
        if results:
            detail_lines = ["*Detailed Bisect Results*\n"]

            # Group by classification for organized display
            by_class = {}
            for r in results:
                cls = r.get("classification", "unknown")
                by_class.setdefault(cls, []).append(r)

            class_display_order = [
                "code_regression",
                "hardware_issue",
                "environment_change",
                "flaky_test",
                "unknown",
            ]

            for cls in class_display_order:
                cls_results = by_class.get(cls, [])
                if not cls_results:
                    continue

                emoji = CLASSIFICATION_EMOJI.get(cls, "⚪")
                label = CLASSIFICATION_LABELS.get(cls, cls)
                detail_lines.append(f"\n*━━━ {emoji} {label} ━━━*\n")

                for r in cls_results:
                    target = r.get("target", {})
                    test_file = target.get("test_file", "unknown")
                    job_name = target.get("job_name", "unknown")
                    confidence = r.get("confidence", "unknown")
                    evidence = r.get("evidence_summary", "N/A")
                    fix = r.get("recommended_fix", "N/A")
                    suspected = r.get("suspected_commit")
                    suspected_pr = r.get("suspected_pr")
                    job_url = target.get("last_failure_job_url", "")
                    streak = target.get("current_streak", 0)

                    detail_lines.append(f"• *`{test_file}`* in `{job_name}`")
                    detail_lines.append(
                        f"  Streak: {streak} | Confidence: {confidence}"
                    )

                    if suspected:
                        cause_str = f"`{suspected}`"
                        if suspected_pr:
                            cause_str += f" (PR #{suspected_pr})"
                        detail_lines.append(f"  Suspected: {cause_str}")

                    detail_lines.append(f"  Evidence: {evidence}")
                    detail_lines.append(f"  Fix: {fix}")

                    if job_url:
                        detail_lines.append(f"  <{job_url}|View failing job>")
                    detail_lines.append("")

            detail_text = "\n".join(detail_lines)

            # Slack has a 4000 char limit per message; split if needed
            if len(detail_text) > 3900:
                chunks = []
                current_chunk = ""
                for line in detail_lines:
                    if len(current_chunk) + len(line) + 1 > 3900:
                        chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        current_chunk += "\n" + line if current_chunk else line
                if current_chunk:
                    chunks.append(current_chunk)

                for chunk in chunks:
                    client.chat_postMessage(
                        channel=SLACK_CHANNEL_ID,
                        thread_ts=thread_ts,
                        text=chunk,
                    )
            else:
                client.chat_postMessage(
                    channel=SLACK_CHANNEL_ID,
                    thread_ts=thread_ts,
                    text=detail_text,
                )

        logger.info("Bisect results posted to Slack successfully")
        return True

    except Exception:
        logger.exception("Failed to post bisect results to Slack")
        return False


def main():
    parser = argparse.ArgumentParser(description="Post CI auto-bisect results to Slack")
    parser.add_argument(
        "--report-file",
        type=str,
        required=True,
        help="Path to bisect_results.json",
    )

    args = parser.parse_args()

    success = post_bisect_to_slack(args.report_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
