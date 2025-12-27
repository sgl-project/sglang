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

        # Map workflow data keys to display names and hardware category
        # Format: (display_name, hardware, test_type_order)
        # test_type_order: 0 = PR Test, 1 = Nightly (so PR Test comes first)
        workflow_info_map = {
            # Nvidia
            "pr_test_nvidia_scheduled_data": ("PR Test", "Nvidia", 0),
            "nightly_nvidia_scheduled_data": ("Nightly", "Nvidia", 1),
            # AMD
            "pr_test_amd_scheduled_data": ("PR Test", "AMD", 0),
            "nightly_amd_scheduled_data": ("Nightly", "AMD", 1),
            # Intel/Xeon
            "pr_test_xeon_scheduled_data": ("PR Test", "Intel", 0),
            "nightly_intel_scheduled_data": ("Nightly", "Intel", 1),
            # XPU
            "pr_test_xpu_scheduled_data": ("PR Test", "XPU", 0),
            # NPU
            "pr_test_npu_scheduled_data": ("PR Test", "NPU", 0),
            "nightly_npu_scheduled_data": ("Nightly", "NPU", 1),
        }

        # Hardware priority order (Nvidia first)
        hardware_order = ["Nvidia", "AMD", "Intel", "XPU", "NPU"]

        # Iterate through each workflow section
        for workflow_key, workflow_data in report_data.items():
            # Skip non-workflow keys (summary, limits, etc.)
            if not isinstance(workflow_data, dict) or not any(
                isinstance(v, dict) and "current_streak" in v
                for v in workflow_data.values()
            ):
                continue

            # Only process scheduled workflows that are in our map
            if workflow_key not in workflow_info_map:
                continue

            test_type, hardware, test_order = workflow_info_map[workflow_key]

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
                            "hardware": hardware,
                            "test_type": test_type,
                            "test_order": test_order,
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

        # Group by hardware, then by test type
        # Structure: {hardware: {test_type: [job_names]}}
        hardware_jobs = {}
        for job in critical_failures:
            hardware = job.get("hardware", "Unknown")
            test_type = job.get("test_type", "Unknown")
            job_name = job.get("job_name", "unknown")
            if hardware not in hardware_jobs:
                hardware_jobs[hardware] = {}
            if test_type not in hardware_jobs[hardware]:
                hardware_jobs[hardware][test_type] = []
            hardware_jobs[hardware][test_type].append(job_name)

        # Create summary message
        workflow_url = ""
        if run_id:
            workflow_url = (
                f"https://github.com/sgl-project/sglang/actions/runs/{run_id}"
            )

        if not hardware_jobs:
            summary = "✅ No critical failures detected in scheduled runs"
            if workflow_url:
                summary += f"\n<{workflow_url}|View CI Monitor Run>"
            color = "good"
        else:
            # Ping relevant people when there are failures
            mentions = "<@U09RR5TNC94> <@U09ABMCKQPM>"
            summary_lines = [f"{mentions} 🚨 *CI Critical Failures (Scheduled Runs)*"]

            # Iterate in hardware priority order, with PR Test before Nightly
            test_type_order = ["PR Test", "Nightly"]
            for hardware in hardware_order:
                if hardware not in hardware_jobs:
                    continue
                summary_lines.append(f"\n*{hardware}:*")
                for test_type in test_type_order:
                    if test_type not in hardware_jobs[hardware]:
                        continue
                    jobs = hardware_jobs[hardware][test_type]
                    job_list = ", ".join(jobs)
                    summary_lines.append(f"  • {test_type}: {job_list}")

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
        if hardware_jobs:
            details_lines = ["*Detailed Failure Breakdown*\n"]

            # Sort critical_failures by hardware order, then test_order
            hardware_order_map = {hw: i for i, hw in enumerate(hardware_order)}
            sorted_failures = sorted(
                critical_failures,
                key=lambda x: (
                    hardware_order_map.get(x.get("hardware", ""), 99),
                    x.get("test_order", 99),
                    x.get("job_name", ""),
                ),
            )

            current_hardware = None
            for job in sorted_failures:
                hardware = job.get("hardware", "Unknown")
                test_type = job.get("test_type", "Unknown")
                job_name = job.get("job_name", "unknown")
                consecutive = job.get("consecutive_failures", 0)
                first_url = job.get("first_failed_url", "")
                first_at = job.get("first_failed_at", "unknown")
                last_url = job.get("last_failed_url", "")
                last_at = job.get("last_failed_at", "unknown")

                # Add hardware section header
                if hardware != current_hardware:
                    details_lines.append(f"\n*━━━ {hardware} ━━━*")
                    current_hardware = hardware

                details_lines.append(
                    f"• *{test_type}* → `{job_name}`\n"
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
