# SPDX-License-Identifier: Apache-2.0

import enum
import logging
import os

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


class ConnectorType(str, enum.Enum):
    FS = "filesystem"
    KV = "KV"
    INSTANCE = "instance"


def create_remote_connector(url, device, **kwargs) -> BaseConnector:
    url = verify_if_url_is_gcs_bucket(url)
    connector_type = parse_connector_type(url)
    if connector_type == "redis":
        return RedisConnector(url)
    elif connector_type == "s3":
        return S3Connector(url)
    elif connector_type == "instance":
        return RemoteInstanceConnector(url, device)
    else:
        raise ValueError(f"Invalid connector type: {url}")

def verify_if_url_is_gcs_bucket(url):
    if url.startswith("gs://"):
        os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = "https://storage.googleapis.com"
        os.environ["AWS_ENDPOINT_URL"] = "https://storage.googleapis.com"
        os.environ["RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"] = "0"
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        url = url.replace("gs://", "s3://", 1)
    return url

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
    "create_remote_connector",
    "get_connector_type",
]
