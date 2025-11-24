import logging
import os
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            f"*GitHub Run ID:* {run_id}\n"
            f"*Case ID:* `{case_id}`\n"
            f"*Model:* `{model}`\n"
            f"*Prompt:* {prompt}"
        )

        client = WebClient(token=token)
        channel_id = "C0A02NDF7UY"
        thread_ts = None

        try:
            history = client.conversations_history(channel=channel_id, limit=10)
            for msg in history.get("messages", []):
                if f"*GitHub Run ID:* {run_id}" in msg.get("text", ""):
                    thread_ts = msg.get("ts")
                    break
        except Exception as e:
            logger.warning(f"Failed to search slack history: {e}")

        client.files_upload_v2(
            channel=channel_id,
            file_uploads=uploads,
            initial_comment=message,
            thread_ts=thread_ts,
        )

        logger.info(f"File uploaded successfully: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        logger.info(f"Slack upload failed: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
