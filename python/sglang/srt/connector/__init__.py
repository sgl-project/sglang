# SPDX-License-Identifier: Apache-2.0

import enum
import logging

from sglang.srt.connector.base_connector import (
    BaseConnector,
    BaseFileSystemConnector,
    BaseKVConnector,
    BaseWeightConnector,
)
from sglang.srt.connector.file import FileConnector
from sglang.srt.connector.redis import RedisConnector
from sglang.srt.connector.s3 import S3Connector
from sglang.srt.utils import parse_connector_type

logger = logging.getLogger(__name__)


class ConnectorType(int, enum.Enum):
    NONE = 0
    KV = 1 << 0
    WEIGHT = 1 << 1
    FS = 1 << 2


def create_remote_connector(url, **kwargs) -> BaseConnector:
    connector_type = parse_connector_type(url)
    if connector_type == "redis":
        return RedisConnector(url)
    elif connector_type == "s3":
        return S3Connector(url)
    elif connector_type == "file":
        return FileConnector(url)
    else:
        raise ValueError(f"Invalid connector type: {url}")


def get_connector_type(client: BaseConnector) -> ConnectorType:
    result = ConnectorType.NONE
    if isinstance(client, BaseKVConnector):
        result |= ConnectorType.KV
    if isinstance(client, BaseWeightConnector):
        result |= ConnectorType.WEIGHT
    if isinstance(client, BaseFileSystemConnector):
        result |= ConnectorType.FS

    return result


__all__ = [
    "BaseConnector",
    "BaseFileSystemConnector",
    "BaseKVConnector",
    "BaseWeightConnector" "ConnectorType",
    "create_remote_connector",
    "get_connector_type",
]
