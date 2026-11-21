"""Configuration for the Google Cloud Storage HiCache backend.

The block comes from ``--hicache-storage-backend-extra-config`` and shares the
JSON object with the factory's own keys, which are ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

DEFAULT_MAX_WORKERS = 16


@dataclass(frozen=True)
class GCSConfig:
    """Connection settings for one GCS bucket."""

    # The bucket must already exist; the backend never creates one.
    bucket: str
    # Object-name prefix, e.g. "hicache/cluster-a". Different deployments may
    # share a bucket as long as their prefixes differ.
    prefix: str = ""
    # Project used for billing and quota. Default: the credential's project.
    project: Optional[str] = None
    # Path to a service-account JSON key. Default: Application Default
    # Credentials (GOOGLE_APPLICATION_CREDENTIALS, workload identity, ...).
    credentials: Optional[str] = None
    max_workers: int = DEFAULT_MAX_WORKERS

    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> GCSConfig:
        names = {field.name for field in fields(cls)}
        values = {k: v for k, v in (extra_config or {}).items() if k in names}
        if not values.get("bucket"):
            raise ValueError(
                "The GCS HiCache backend requires a non-empty 'bucket' in "
                "--hicache-storage-backend-extra-config."
            )
        config = cls(**values)
        if config.max_workers < 1:
            raise ValueError(
                f"GCS max_workers must be at least 1, got {config.max_workers}."
            )
        return config
