#!/usr/bin/env python3
"""
Post CI failure analysis results to Slack.

This is a standalone script that doesn't depend on sglang package installation.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def post_ci_failures_to_slack(report_file: str) -> bool:
    """
    Post CI failure report to Slack with threaded details.

    Creates a parent message with summary (workflow: job1, job2, ...)
    and a threaded reply with detailed failure information.

    Args:
        report_file: Path to JSON file containing failure analysis from ci_failures_analysis.py

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from slack_sdk import WebClient

        token = os.environ.get("SGLANG_DIFFUSION_SLACK_TOKEN")
        if not token:
            logger.info("Slack post failed: no token")
            return False

        # CI failures channel
        channel_id = "C0A2DG0R7CJ"

        # Get GitHub run ID for linking to the workflow run
        run_id = os.environ.get("GITHUB_RUN_ID", "")

        # Load report data
        with open(report_file, "r") as f:
            report_data = json.load(f)

        client = WebClient(token=token)

        # Parse the real JSON structure
        # The JSON has workflow sections like "pr_test_nvidia_scheduled_data", "nightly_scheduled_data"
        # Each section contains jobs with their stats including "current_streak"

        critical_failures = []

        # Map workflow data keys to display names
        workflow_name_map = {
            # PR Tests - Scheduled (5 workflows)
            "pr_test_nvidia_scheduled_data": "PR Test (Nvidia, scheduled)",
            "pr_test_amd_scheduled_data": "PR Test (AMD, scheduled)",
            "pr_test_xeon_scheduled_data": "PR Test (Xeon, scheduled)",
            "pr_test_xpu_scheduled_data": "PR Test (XPU, scheduled)",
            "pr_test_npu_scheduled_data": "PR Test (NPU, scheduled)",
            # Nightly Tests - Scheduled (4 workflows)
            "nightly_nvidia_scheduled_data": "Nightly Test (Nvidia, scheduled)",
            "nightly_amd_scheduled_data": "Nightly Test (AMD, scheduled)",
            "nightly_intel_scheduled_data": "Nightly Test (Intel, scheduled)",
            "nightly_npu_scheduled_data": "Nightly Test (NPU, scheduled)",
        }

        # Iterate through each workflow section
        for workflow_key, workflow_data in report_data.items():
            # Skip non-workflow keys (summary, limits, etc.)
            if not isinstance(workflow_data, dict) or not any(
                isinstance(v, dict) and "current_streak" in v
                for v in workflow_data.values()
            ):
                continue

            # Get workflow display name
            workflow_name = workflow_name_map.get(workflow_key, workflow_key)

            # Only process scheduled workflows
            if "scheduled" not in workflow_key.lower():
                continue

            # Check each job in this workflow
            for job_name, job_data in workflow_data.items():
                if not isinstance(job_data, dict):
                    continue

                current_streak = job_data.get("current_streak", 0)

                # Filter for jobs with streak >= 2
                if current_streak >= 2:
                    first_failure = job_data.get("first_failure_in_streak", {})
                    last_failure = job_data.get("last_failure_in_streak", {})

                    critical_failures.append(
                        {
                            "workflow_name": workflow_name,
                            "job_name": job_name,
                            "consecutive_failures": current_streak,
                            "first_failed_at": (
                                first_failure.get("created_at", "unknown")
                                if first_failure
                                else "unknown"
                            ),
                            "first_failed_url": (
                                first_failure.get("job_url", "")
                                if first_failure
                                else ""
                            ),
                            "last_failed_at": (
                                last_failure.get("created_at", "unknown")
                                if last_failure
                                else "unknown"
                            ),
                            "last_failed_url": (
                                last_failure.get("job_url", "") if last_failure else ""
                            ),
                        }
                    )

        # Group by workflow
        workflow_jobs = {}
        for job in critical_failures:
            workflow = job.get("workflow_name", "Unknown")
            job_name = job.get("job_name", "unknown")
            if workflow not in workflow_jobs:
                workflow_jobs[workflow] = []
            workflow_jobs[workflow].append(job_name)

        # Create summary message
        workflow_url = ""
        if run_id:
            workflow_url = (
                f"https://github.com/sgl-project/sglang/actions/runs/{run_id}"
            )

        if not workflow_jobs:
            summary = "✅ No critical failures detected in scheduled runs"
            if workflow_url:
                summary += f"\n<{workflow_url}|View CI Monitor Run>"
            color = "good"
        else:
            # Ping relevant people when there are failures
            mentions = "<@U09RR5TNC94> <@U09ABMCKQPM>"
            summary_lines = [f"{mentions} 🚨 *CI Critical Failures (Scheduled Runs)*"]
            for workflow, jobs in sorted(workflow_jobs.items()):
                job_list = ", ".join(jobs)
                summary_lines.append(f"• *{workflow}*: {job_list}")
            if workflow_url:
                summary_lines.append(f"\n<{workflow_url}|View Full CI Monitor Report>")
            summary = "\n".join(summary_lines)
            color = "danger"

        # Post parent message
        response = client.chat_postMessage(
            channel=channel_id,
            text=summary,
            attachments=[
                {
                    "color": color,
                    "footer": "SGLang CI Monitor",
                    "footer_icon": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                    "ts": int(datetime.now().timestamp()),
                }
            ],
        )

        thread_ts = response["ts"]

        # If there are failures, post detailed breakdown in thread
        if workflow_jobs:
            details_lines = ["*Detailed Failure Breakdown*\n"]

            for job in critical_failures:
                workflow = job.get("workflow_name", "Unknown")
                job_name = job.get("job_name", "unknown")
                consecutive = job.get("consecutive_failures", 0)
                first_url = job.get("first_failed_url", "")
                first_at = job.get("first_failed_at", "unknown")
                last_url = job.get("last_failed_url", "")
                last_at = job.get("last_failed_at", "unknown")

                details_lines.append(
                    f"• *{workflow}* → `{job_name}`\n"
                    f"  Consecutive failures: {consecutive}\n"
                    f"  First failed: <{first_url}|{first_at}>\n"
                    f"  Last failed: <{last_url}|{last_at}>\n"
                )

            details_text = "\n".join(details_lines)

            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=details_text,
            )

        logger.info("CI failure report posted to Slack successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to post CI failures to Slack: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Post CI failure analysis results to Slack"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        required=True,
        help="Path to CI failure analysis JSON report",
    )

    args = parser.parse_args()

    success = post_ci_failures_to_slack(args.report_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
