# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/runai_utils.py

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_BOTO3_S3_UNSIGNED_PATCH_LOCK = threading.Lock()
_BOTO3_S3_UNSIGNED_PATCH_APPLIED = False


def _env_nonempty(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and str(v).strip() != ""


def _aws_shared_credentials_file_has_static_keys() -> bool:
    """True if the shared credentials file looks like it defines access keys."""
    path = os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or os.path.expanduser(
        "~/.aws/credentials"
    )
    try:
        if not os.path.isfile(path):
            return False
        with open(path, encoding="utf-8", errors="ignore") as f:
            body = f.read()
    except OSError:
        return False
    lower = body.lower()
    return "[" in body and (
        "aws_access_key_id" in lower or "aws_secret_access_key" in lower
    )


def _has_standard_aws_credentials() -> bool:
    """Whether boto3 is likely to resolve credentials without UNSIGNED (heuristic)."""
    if _env_nonempty("AWS_ACCESS_KEY_ID") and _env_nonempty("AWS_SECRET_ACCESS_KEY"):
        return True
    if _env_nonempty("AWS_WEB_IDENTITY_TOKEN_FILE") and _env_nonempty("AWS_ROLE_ARN"):
        return True
    if _env_nonempty("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or _env_nonempty(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI"
    ):
        return True
    if _aws_shared_credentials_file_has_static_keys():
        return True
    return False


def _ec2_metadata_disabled() -> bool:
    """EC2 instance metadata service disabled (common for local MinIO / custom S3)."""
    v = os.environ.get("AWS_EC2_METADATA_DISABLED", "").strip().lower()
    return v in ("true", "1", "yes")


def _should_use_unsigned_s3_for_runai() -> bool:
    """Use UNSIGNED S3 only when there are no explicit creds and IMDS is off.

    If IMDS is left enabled, boto3 may still obtain an IAM role; forcing UNSIGNED
    would break that path. Typical public-bucket + MinIO setup: no keys on disk,
    ``AWS_EC2_METADATA_DISABLED=true``.
    """
    if _has_standard_aws_credentials():
        return False
    if not _ec2_metadata_disabled():
        return False
    return True


def _install_boto3_s3_unsigned_patch(boto3: Any, Config: Any, UNSIGNED: Any) -> None:
    """Merge UNSIGNED into config for boto3 S3 clients only (idempotent install)."""

    def _merge_unsigned(cfg: Any) -> Any:
        unsigned_cfg = Config(signature_version=UNSIGNED)
        if cfg is None:
            return unsigned_cfg
        return cfg.merge(unsigned_cfg)

    _orig_client = boto3.client
    _orig_session_client = boto3.session.Session.client

    def _client(*args: Any, **kwargs: Any) -> Any:
        if args and args[0] == "s3":
            kwargs = dict(kwargs)
            kwargs["config"] = _merge_unsigned(kwargs.get("config"))
        return _orig_client(*args, **kwargs)

    def _session_client(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args and args[0] == "s3":
            kwargs = dict(kwargs)
            kwargs["config"] = _merge_unsigned(kwargs.get("config"))
        return _orig_session_client(self, *args, **kwargs)

    boto3.client = _client
    boto3.session.Session.client = _session_client


def _maybe_patch_boto3_s3_unsigned() -> None:
    """Patch boto3 so new S3 clients use UNSIGNED when appropriate.

    Enabled when no standard AWS credentials are configured and
    ``AWS_EC2_METADATA_DISABLED`` is true (anonymous / public buckets, MinIO, etc.).
    Safe to call multiple times; only S3 clients are affected.
    """
    if not _should_use_unsigned_s3_for_runai():
        return

    global _BOTO3_S3_UNSIGNED_PATCH_APPLIED

    with _BOTO3_S3_UNSIGNED_PATCH_LOCK:
        if _BOTO3_S3_UNSIGNED_PATCH_APPLIED:
            return
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError as e:
            logger.warning(
                "RunAI unsigned S3: boto3/botocore not available (%s); "
                "install boto3 for S3 object storage, or provide AWS credentials",
                e,
            )
            return

        _install_boto3_s3_unsigned_patch(boto3, Config, UNSIGNED)
        _BOTO3_S3_UNSIGNED_PATCH_APPLIED = True
        logger.info(
            "RunAI object storage: S3 clients use signature_version=UNSIGNED "
            "(no static credentials, AWS_EC2_METADATA_DISABLED)"
        )


SUPPORTED_SCHEMES = ["s3://", "gs://", "az://"]

# Design Pattern: Single Metadata Download Before Process Launch

#   1. Engine entrypoint (engine.py) or server arguments post init  (server_args.py):
#     - Downloads config/tokenizer metadata ONCE before launching subprocesses
#     - This happens in the main process, avoiding multi-process coordination
#
#   2. ModelConfig/HF Utils (model_config.py, hf_transformers_utils.py):
#     - Use ObjectStorageModel.get_path() to retrieve the cached local path
#     - NO re-download - just path resolution
#
#   3. RunaiModelStreamerLoader (loader.py):
#     - Calls list_safetensors() which operates directly on the object storage URI
#     - Streams weights lazily during model loading

#   This avoids file locks, race conditions, and duplicate downloads


def list_safetensors(path: str = "") -> list[str]:
    """
    List full file names from object path and filter by allow pattern.

    Args:
        path: The object storage path to list from.

    Returns:
        list[str]: List of full object storage paths allowed by the pattern
    """
    _maybe_patch_boto3_s3_unsigned()
    from runai_model_streamer import list_safetensors as runai_list_safetensors

    return runai_list_safetensors(path)


def is_runai_obj_uri(model_or_path: str | Path) -> bool:
    # Cast to str to handle pathlib.Path inputs which lack string methods (like .lower)
    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))


class ObjectStorageModel:
    """
    Model loader that uses Runai Model Streamer to load a model.

      Supports object storage (S3, GCS) with lazy weight streaming.

      Configuration (via load_config.model_loader_extra_config):
          - distributed (bool): Enable distributed streaming
          - concurrency (int): Number of concurrent downloads
          - memory_limit (int): Memory limit for streaming buffer

      Note: Metadata files must be pre-downloaded via
      ObjectStorageModel.download_and_get_path() before instantiation.

    Attributes:
        dir: The temporary created directory.
    """

    def __init__(self, url: str) -> None:
        _maybe_patch_boto3_s3_unsigned()
        self.dir = ObjectStorageModel.get_path(url)

        from runai_model_streamer import ObjectStorageModel as RunaiObjectStorageModel

        self._runai_obj = RunaiObjectStorageModel(model_path=url, dst=self.dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._runai_obj.__exit__(exc_type, exc_val, exc_tb)

    def pull_files(
        self,
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:
        """Pull files from object storage into the local cache directory.

        Args:
            allow_pattern: File patterns to include (e.g. ["*.json"]).
            ignore_pattern: File patterns to exclude.
        """
        self._runai_obj.pull_files(allow_pattern, ignore_pattern)

    @classmethod
    def download_and_get_path(cls, model_path: str) -> str:
        """
        Downloads the model metadata (excluding heavy weights) and returns
        the local directory path. Safe for concurrent usage by multiple processes
        """
        with cls(url=model_path) as downloader:
            downloader.pull_files(
                ignore_pattern=[
                    "*.pt",
                    "*.safetensors",
                    "*.bin",
                    "*.tensors",
                    "*.pth",
                ],
            )
            cache_dir = downloader.dir
            logger.info(f"Runai Model : {cache_dir}, metadata ready.")
        return cache_dir

    @classmethod
    def get_path(cls, model_path: str) -> str:
        """
        Returns the local directory path.
        """
        model_hash = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]
        base_dir = envs.SGLANG_CACHE_DIR.get()

        # Ensure base cache dir exists
        os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)

        return os.path.join(
            base_dir,
            "model_streamer",
            model_hash,
        )
