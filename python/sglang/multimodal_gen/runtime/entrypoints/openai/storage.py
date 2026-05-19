import asyncio
import os
from typing import Optional

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


CONTENT_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".glb": "model/gltf-binary",
    ".obj": "text/plain",
}

DEFAULT_UPLOAD_TIMEOUT = 60


def get_content_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return CONTENT_TYPE_MAP.get(ext, "application/octet-stream")


class CloudStorage:
    def __init__(self):
        self.upload_timeout = float(
            os.getenv("SGLANG_S3_UPLOAD_TIMEOUT", DEFAULT_UPLOAD_TIMEOUT)
        )
        self.enabled = os.getenv("SGLANG_CLOUD_STORAGE_TYPE", "").lower() == "s3"
        if not self.enabled:
            return

        try:
            import boto3
            from botocore.config import Config
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

        boto_config = Config(
            connect_timeout=60,
            read_timeout=self.upload_timeout,
        )
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("SGLANG_S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SGLANG_S3_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=boto_config,
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
            content_type = get_content_type(local_path)
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

    async def upload_and_cleanup(
        self, file_path: str, presigned_url: Optional[str] = None
    ) -> Optional[str]:
        if presigned_url:
            success = await self.upload_to_presigned_url(
                presigned_url, file_path, cleanup=True
            )
            return presigned_url if success else None

        if not self.is_enabled():
            return None

        key = os.path.basename(file_path)
        url = await self.upload_file(file_path, key)

        if url:
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        return url

    async def upload_to_presigned_url(
        self,
        presigned_url: str,
        file_path: str,
        cleanup: bool = False,
    ) -> bool:
        try:
            import httpx
        except ImportError:
            logger.error(
                "httpx is not installed. Please install it with `pip install httpx` "
                "to use presigned URL upload."
            )
            return False

        content_type = get_content_type(file_path)

        def _sync_upload() -> int:
            """Synchronous upload in thread executor to avoid blocking event loop."""
            with httpx.Client(timeout=self.upload_timeout) as client:
                with open(file_path, "rb") as f:
                    response = client.put(
                        presigned_url,
                        content=f,
                        headers={"Content-Type": content_type},
                    )
                    return response.status_code

        try:
            status_code = await asyncio.get_running_loop().run_in_executor(
                None, _sync_upload
            )
            if 200 <= status_code < 300:
                logger.info(f"Uploaded {file_path} to presigned URL")
                if cleanup:
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove temporary file {file_path}: {e}"
                        )
                return True
            else:
                logger.error(f"Upload to presigned URL failed: {status_code}")
                return False
        except IOError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Upload to presigned URL failed: {e}")
            return False


# Global instance
cloud_storage = CloudStorage()
