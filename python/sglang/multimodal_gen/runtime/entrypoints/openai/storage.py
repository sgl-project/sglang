import asyncio
import os
from typing import Optional

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CloudStorage:
    def __init__(self):
        self.enabled = os.getenv("SGLANG_CLOUD_STORAGE_TYPE", "").lower() == "s3"
        if not self.enabled:
            return

        try:
            import boto3
        except ImportError:
            logger.error(
                "boto3 is not installed. Please install it with `pip install boto3` to use cloud storage."
            )
            self.enabled = False
            return

        self.bucket_name = os.getenv("SGLANG_S3_BUCKET_NAME")
        if not self.bucket_name:
            self.enabled = False
            return

        endpoint_url = os.getenv("SGLANG_S3_ENDPOINT_URL") or None
        region_name = os.getenv("SGLANG_S3_REGION_NAME") or None

        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("SGLANG_S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SGLANG_S3_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
        self.endpoint_url = endpoint_url
        self.region_name = region_name

    def is_enabled(self) -> bool:
        return self.enabled

    async def upload_file(self, local_path: str, destination_key: str) -> Optional[str]:
        if not self.is_enabled():
            return None

        def _sync_upload():
            """Synchronous part of the upload to run in a thread."""
            ext = os.path.splitext(local_path)[1].lower()
            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp4": "video/mp4",
            }.get(ext, "application/octet-stream")

            # Use the client created once in __init__
            self.client.upload_file(
                local_path,
                self.bucket_name,
                destination_key,
                ExtraArgs={"ContentType": content_type},
            )

        try:
            # Offload the blocking I/O call to a thread executor
            await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
        except Exception as e:
            # If upload fails, log the error and return None for fallback
            logger.error(f"Upload failed for {destination_key}: {e}")
            return None

        # Simplified URL generation with a default region
        if self.endpoint_url:
            url = (
                f"{self.endpoint_url.rstrip('/')}/{self.bucket_name}/{destination_key}"
            )
        else:
            region = self.region_name or "us-east-1"
            url = f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{destination_key}"

        logger.info(f"Uploaded {local_path} to {url}")
        return url

    async def upload_and_cleanup(self, file_path: str) -> Optional[str]:
        """Helper to upload a file and delete the local copy if successful."""
        if not self.is_enabled():
            return None

        key = os.path.basename(file_path)
        url = await self.upload_file(file_path, key)

        if url:
            try:
                # pass if removal fails
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        return url


# Global instance
cloud_storage = CloudStorage()
