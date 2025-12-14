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
        channel_id = "C09HCG2HM1T"

        # Load report data
        with open(report_file, "r") as f:
            report_data = json.load(f)

        client = WebClient(token=token)

        # Build summary by workflow
        failing_jobs = report_data.get("failing_jobs", [])
        critical_failures = [
            job
            for job in failing_jobs
            if job.get("consecutive_failures", 0) >= 2
            and "scheduled" in job.get("workflow_name", "").lower()
        ]

        # Group by workflow
        workflow_jobs = {}
        for job in critical_failures:
            workflow = job.get("workflow_name", "Unknown")
            job_name = job.get("job_name", "unknown")
            if workflow not in workflow_jobs:
                workflow_jobs[workflow] = []
            workflow_jobs[workflow].append(job_name)

        # Create summary message
        if not workflow_jobs:
            summary = "✅ No critical failures detected in scheduled runs"
            color = "good"
        else:
            # Ping relevant people when there are failures
            mentions = "<@U09RR5TNC94>"
            summary_lines = [
                f"{mentions} 🚨 *CI Critical Failures (Scheduled Runs)*",
                "_Note: Recent runs are shown left to right in the detailed breakdown_\n",
            ]
            for workflow, jobs in sorted(workflow_jobs.items()):
                job_list = ", ".join(jobs)
                summary_lines.append(f"• *{workflow}*: {job_list}")
            summary = "\n".join(summary_lines)
            color = "danger"

        # Post parent message
        response = client.chat_postMessage(
            channel=channel_id,
            text=summary,
            username="SGLang CI Bot",
            icon_emoji=":robot_face:",
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
                username="SGLang CI Bot",
                icon_emoji=":robot_face:",
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
