"""
    This file upload the media generated in diffusion-nightly-test to a slack channel of SGLang
"""

import logging
import os
import tempfile
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import urlopen

from sglang.multimodal_gen.runtime.utils.perf_logger import get_git_commit_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import inspect

try:
    import sglang.multimodal_gen.test.server.testcase_configs as configs
    from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

    ALL_CASES = []
    for name, value in inspect.getmembers(configs):
        if name.endswith("_CASES") or "_CASES_" in name:
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], DiffusionTestCase)
            ):
                ALL_CASES.extend(value)
            elif isinstance(value, list) and len(value) == 0:
                # Assume empty list with matching name is a valid case list container
                pass

    # Deduplicate cases by ID
    seen_ids = set()
    unique_cases = []
    for c in ALL_CASES:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            unique_cases.append(c)
    ALL_CASES = unique_cases

except Exception as e:
    logger.warning(f"Failed to import test cases: {e}")
    ALL_CASES = []


def _get_status_message(run_id, current_case_id, thread_messages=None):
    date_str = datetime.now().strftime("%d/%m")
    base_header = f"""🧵 for nightly test of {date_str}*
*Git Revision:* {get_git_commit_hash()}
*GitHub Run ID:* {run_id}
*Total Tasks:* {len(ALL_CASES)}

"""

    if not ALL_CASES:
        return base_header

    default_emoji_for_case_in_progress = "⏳"
    status_map = {c.id: default_emoji_for_case_in_progress for c in ALL_CASES}

    if thread_messages:
        for msg in thread_messages:
            text = msg.get("text", "")
            # Look for case_id in the message (format: *Case ID:* `case_id`)
            for c in ALL_CASES:
                if f"*Case ID:* `{c.id}`" in text:
                    status_map[c.id] = "✅"

    if current_case_id:
        status_map[current_case_id] = "✅"

    lines = [base_header, "", "*Tasks Status:*"]

    # Calculate padding
    max_len = max(len(c.id) for c in ALL_CASES) if ALL_CASES else 10
    max_len = max(max_len, len("Case ID"))

    # Build markdown table inside a code block
    table_lines = ["```"]
    table_lines.append(f"| {'Case ID'.ljust(max_len)} | Status |")
    table_lines.append(f"| {'-' * max_len} | :----: |")

    for c in ALL_CASES:
        mark = status_map.get(c.id, default_emoji_for_case_in_progress)
        table_lines.append(f"| {c.id.ljust(max_len)} |   {mark}   |")

    table_lines.append("```")

    lines.extend(table_lines)

    return "\n".join(lines)


def upload_file_to_slack(
    case_id: str = None,
    model: str = None,
    prompt: str = None,
    file_path: str = None,
    origin_file_path: str = None,
) -> bool:
    temp_path = None
    try:
        from slack_sdk import WebClient

        run_id = os.getenv("GITHUB_RUN_ID", "local")

        token = os.environ.get("SGLANG_DIFFUSION_SLACK_TOKEN")
        if not token:
            logger.info(f"Slack upload failed: no token")
            return False

        if not file_path or not os.path.exists(file_path):
            logger.info(f"Slack upload failed: no file path")
            return False

        if origin_file_path and origin_file_path.startswith(("http", "https")):
            suffix = os.path.splitext(urlparse(origin_file_path).path)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                with urlopen(origin_file_path) as response:
                    tf.write(response.read())
                temp_path = tf.name
                origin_file_path = temp_path

        uploads = [{"file": file_path, "title": "Generated Image"}]
        if origin_file_path and os.path.exists(origin_file_path):
            uploads.insert(0, {"file": origin_file_path, "title": "Original Image"})

        message = (
            f"*Case ID:* `{case_id}`\n" f"*Model:* `{model}`\n" f"*Prompt:* {prompt}"
        )

        client = WebClient(token=token)
        channel_id = "C0A02NDF7UY"
        thread_ts = None

        parent_msg_text = None
        try:
            history = client.conversations_history(channel=channel_id, limit=100)
            for msg in history.get("messages", []):
                if f"*GitHub Run ID:* {run_id}" in msg.get("text", ""):
                    # Use thread_ts if it exists (msg is a reply), otherwise use ts (msg is a parent)
                    thread_ts = msg.get("thread_ts") or msg.get("ts")
                    parent_msg_text = msg.get("text", "")
                    logger.info(f"Found thread_ts: {thread_ts}")
                    break
        except Exception as e:
            logger.warning(f"Failed to search slack history: {e}")

        if not thread_ts:
            try:
                text = _get_status_message(run_id, case_id)
                response = client.chat_postMessage(channel=channel_id, text=text)
                thread_ts = response["ts"]
            except Exception as e:
                logger.warning(f"Failed to create parent thread: {e}")

        # Upload first to ensure it's in history
        client.files_upload_v2(
            channel=channel_id,
            file_uploads=uploads,
            initial_comment=message,
            thread_ts=thread_ts,
        )

        # Then update status based on thread replies
        if thread_ts:
            try:
                replies = client.conversations_replies(
                    channel=channel_id, ts=thread_ts, limit=200
                )
                messages = replies.get("messages", [])
                new_text = _get_status_message(run_id, case_id, messages)

                # Only update if changed significantly (ignoring timestamp diffs if any)
                # But here we just check text content
                if new_text != parent_msg_text:
                    client.chat_update(channel=channel_id, ts=thread_ts, text=new_text)
            except Exception as e:
                logger.warning(f"Failed to update parent message: {e}")

        logger.info(f"File uploaded successfully: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        logger.info(f"Slack upload failed: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
        import json

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
            mentions = "<@dougyster>"
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
                channel=channel_id, thread_ts=thread_ts, text=details_text
            )

        logger.info(f"CI failure report posted to Slack successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to post CI failures to Slack: {e}")
        return False


if __name__ == "__main__":
    import argparse
    import sys

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
