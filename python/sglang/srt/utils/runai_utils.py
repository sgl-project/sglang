# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/runai_utils.py

import hashlib
import logging
import os
import shutil

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMES = ["s3://", "gs://"]


def get_cache_dir() -> str:
    # Expand user path (~) to ensure absolute paths for locking
    path = os.getenv("SGLANG_CACHE_DIR", "~/.cache/sglang/")
    return os.path.expanduser(path)


def list_safetensors(path: str = "") -> list[str]:
    """
    List full file names from object path and filter by allow pattern.

    Args:
        path: The object storage path to list from.

    Returns:
        list[str]: List of full object storage paths allowed by the pattern
    """
    from runai_model_streamer import list_safetensors as runai_list_safetensors

    return runai_list_safetensors(path)


def is_runai_obj_uri(model_or_path: str) -> bool:
    return model_or_path.lower().startswith(tuple(SUPPORTED_SCHEMES))


class ObjectStorageModel:
    """
    A class representing an ObjectStorage model mirrored into a
    temporary directory.

    Attributes:
        dir: The temporary created directory.

    Methods:
        pull_files(): Pull model from object storage to the temporary directory.
    """

    def __init__(self, url: str) -> None:
        # We moved the directory creation logic partly to __init__ but
        # we must be careful not to delete it if another process is using it.
        # This hash is unique to the URL.
        self.model_hash = hashlib.sha256(str(url).encode()).hexdigest()[:8]
        base_dir = get_cache_dir()

        # Ensure base cache dir exists
        os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)

        self.dir = os.path.join(
            base_dir,
            "model_streamer",
            self.model_hash,
        )

    def pull_files(
        self,
        model_path: str = "",
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:
        """
        Pull files from object storage into the temporary directory.
        Safe for concurrent execution using FileLock.
        """
        from runai_model_streamer import pull_files as runai_pull_files

        # Remove the directory if it exists
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)

        # Create the directory
        os.makedirs(self.dir, exist_ok=True)

        if not model_path.endswith("/"):
            model_path = model_path + "/"

        try:
            runai_pull_files(model_path, self.dir, allow_pattern, ignore_pattern)
            # Create a marker file to indicate success?
            # Path(os.path.join(self.dir, ".success")).touch()
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # cleanup partial download
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir)
            raise e

        logger.debug(f"Runai Model hash: {self.model_hash}, download complete.")

    @classmethod
    def download_and_get_path(cls, model_path: str) -> str:
        """
        Downloads the model metadata (excluding heavy weights) and returns
        the local directory path. Safe for concurrent usage.
        """
        # 1. Create the object
        downloader = cls(url=model_path)

        # 2. Pull files with the specific ignore patterns for weights
        downloader.pull_files(
            model_path=model_path,
            ignore_pattern=[
                "*.pt",
                "*.safetensors",
                "*.bin",
                "*.tensors",
                "*.pth",
            ],
        )

        # 3. Return the local directory path
        return downloader.dir

    @classmethod
    def get_path(cls, model_path: str) -> str:
        """
        Returns the local directory path.
        """
        downloader = cls(url=model_path)
        return downloader.dir
