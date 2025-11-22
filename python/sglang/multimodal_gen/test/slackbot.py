import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_file_to_slack(
    file_path: str,
    title: str = None,
    message: str = None,
) -> bool:
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError as e:
        logger.warning(f"Failed to import slack_sdk: {str(e)}. Skipping Slack upload.")
        return False

    token = os.environ.get("SGLANG_DIFFUSION_SLACK_TOKEN")
    channel_id = "C0A02NDF7UY"

    if not token:
        logger.warning("SGLANG_DIFFUSION_SLACK_TOKEN not found. Skipping Slack upload.")
        return False

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    client = WebClient(token=token)

    try:
        logger.info(f"Uploading file to Slack: {file_path}")

        client.files_upload_v2(
            channel=channel_id,
            file=file_path,
            title=title if title else os.path.basename(file_path),
            initial_comment=message,
        )

        logger.info(f"File uploaded successfully: {os.path.basename(file_path)}")
        return True

    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        return False
    except Exception as e:
        logger.error(f"Unknown error during Slack upload: {str(e)}")
        return False
