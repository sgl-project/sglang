# SPDX-License-Identifier: Apache-2.0

import enum
import logging

from sglang.srt.connector.base_connector import (
    BaseConnector,
    BaseFileConnector,
    BaseKVConnector,
)
from sglang.srt.connector.redis import RedisConnector
from sglang.srt.connector.remote_instance import RemoteInstanceConnector
from sglang.srt.connector.s3 import S3Connector
from sglang.srt.utils import parse_connector_type

logger = logging.getLogger(__name__)

# Scheme of HttpRangeConnector, kept here so create_remote_connector can
# dispatch on it without importing the module.
HTTP_RANGE_SCHEME = "http-range"


class ConnectorType(str, enum.Enum):
    FS = "filesystem"
    KV = "KV"
    INSTANCE = "instance"


def _is_azure_blob_url(url: str, connector_type: str) -> bool:
    """Detect Azure Blob Storage URLs.

    Matches ``az://...`` URLs and ``https://<account>.blob.core.windows.net/...``
    URLs, which are the two forms accepted by the ``blobfile`` library.
    """
    if connector_type == "az":
        return True
    return connector_type == "https" and ".blob.core.windows.net" in url


def create_remote_connector(url, device=None, **kwargs) -> BaseConnector:
    connector_type = parse_connector_type(url)
    if connector_type == "redis":
        return RedisConnector(url)
    elif connector_type == "s3":
        return S3Connector(url)
    elif connector_type == "instance":
        return RemoteInstanceConnector(url, device)
    elif connector_type == HTTP_RANGE_SCHEME:
        # Imported lazily to keep the HTTP streaming machinery out of the
        # import path of every other connector.
        from sglang.srt.connector.http_range import HttpRangeConnector

        return HttpRangeConnector(url)
    elif _is_azure_blob_url(url, connector_type):
        # Imported lazily so the optional ``blobfile`` dependency is only
        # required when an Azure URL is actually used.
        from sglang.srt.connector.azure import AzureBlobConnector

        return AzureBlobConnector(url)
    else:
        raise ValueError(f"Invalid connector type: {url}")


def get_connector_type(client: BaseConnector) -> ConnectorType:
    if isinstance(client, BaseKVConnector):
        return ConnectorType.KV
    if isinstance(client, BaseFileConnector):
        return ConnectorType.FS
    if isinstance(client, RemoteInstanceConnector):
        return ConnectorType.INSTANCE

    raise ValueError(f"Invalid connector type: {client}")


__all__ = [
    "BaseConnector",
    "BaseFileConnector",
    "BaseKVConnector",
    "RedisConnector",
    "RemoteInstanceConnector",
    "S3Connector",
    "ConnectorType",
    "HTTP_RANGE_SCHEME",
    "create_remote_connector",
    "get_connector_type",
]
