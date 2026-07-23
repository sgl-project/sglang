# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Any, Final, Generator, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch

from sglang.srt.connector import BaseFileConnector
from sglang.version import __version__ as _sglang_version

_S3_QUERY_KWARGS: Final[Tuple[str, ...]] = (
    "endpoint_url",
    "region_name",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
)

# Set form of ``_S3_QUERY_KWARGS`` for membership / diff checks. Built once
# at module load so per-call hot paths don't reconstruct it.
_S3_QUERY_KWARGS_SET: Final[frozenset[str]] = frozenset(_S3_QUERY_KWARGS)

# Single source of truth for kwargs ``create_remote_connector`` may forward
# to ``S3Connector``. Adds ``config`` (programmatic only, not a query param)
# to the URI keys above.
S3_FORWARDED_KWARGS: Final[frozenset[str]] = _S3_QUERY_KWARGS_SET | frozenset(
    {"config"}
)

# Cap how many unknown query keys are echoed (the rest collapse to "(+N more)").
_MAX_UNKNOWN_KEYS_SHOWN: Final[int] = 10


def _sanitize_query_key(key: str, *, max_len: int = 64) -> str:
    """Make an untrusted query key safe to echo into an error message/log.

    ``max_len`` bounds the *returned* length: a longer key is truncated so the
    result (including the trailing ``...`` marker) is at most ``max_len`` chars.
    """
    cleaned = "".join(ch if ch.isprintable() else "?" for ch in key)
    if len(cleaned) > max_len:
        marker = "..."
        if max_len <= len(marker):
            cleaned = marker[:max_len]
        else:
            cleaned = cleaned[: max_len - len(marker)] + marker
    return cleaned


def _parse_s3_kwargs(url: str) -> tuple[str, dict[str, Any]]:
    """Strip query string from s3 URL and return (clean_url, boto3_kwargs).

    Recognizes ``endpoint_url``, ``region_name``, ``aws_access_key_id``,
    ``aws_secret_access_key``, ``aws_session_token`` from the query string.
    The keys match boto3 verbatim; aliases are not supported.

    Environment variables (``AWS_ENDPOINT_URL``, ``AWS_DEFAULT_REGION``,
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``)
    are resolved by boto3's default chain when no query value is given. The
    helper does not re-inject them as kwargs so they don't override an
    explicit ``botocore.config.Config(region_name=...)`` passed via ``config=``.

    Unknown query keys raise ``ValueError`` rather than being silently
    stripped, so typos like ``?endpiont_url=...`` surface immediately
    instead of failing later as a credentials / endpoint mismatch.
    Duplicate occurrences of a recognized key, malformed fragments
    (``?endpoint_url`` with no ``=``), and blank values
    (``?endpoint_url=``) raise ``ValueError`` for the same reason.

    The returned dict is typed ``dict[str, Any]`` because callers merge
    additional ``boto3.client`` kwargs (e.g. ``botocore.config.Config``) into
    it; the helper itself only emits ``str`` values.
    """
    parsed = urlparse(url)
    try:
        parsed_query = parse_qs(
            parsed.query, strict_parsing=True, keep_blank_values=True
        )
    except ValueError:
        raise ValueError("Malformed s3 URL query string") from None
    unknown = sorted(set(parsed_query) - _S3_QUERY_KWARGS_SET)
    if unknown:
        # Keys are caller-controlled and may be logged; sanitize each and cap how
        # many are listed so a URL with many junk keys can't amplify the error.
        shown = [_sanitize_query_key(key) for key in unknown[:_MAX_UNKNOWN_KEYS_SHOWN]]
        if len(unknown) > _MAX_UNKNOWN_KEYS_SHOWN:
            shown.append(f"(+{len(unknown) - _MAX_UNKNOWN_KEYS_SHOWN} more)")
        raise ValueError(
            f"Unknown s3 URL query parameter(s): {', '.join(shown)}; "
            f"allowed: {', '.join(_S3_QUERY_KWARGS)}"
        )
    kwargs: dict[str, Any] = {}
    for key in _S3_QUERY_KWARGS:
        values = parsed_query.get(key)
        if not values:
            continue
        if len(values) > 1:
            raise ValueError(
                f"Duplicate query parameter {key!r} in s3 URL; expected one value"
            )
        if values[0] == "":
            raise ValueError(f"Empty value for query parameter {key!r} in s3 URL")
        kwargs[key] = values[0]
    clean = parsed._replace(query="", fragment="").geturl()
    return clean, kwargs


def _filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def list_files(
    s3,
    path: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:
    """
    List files from S3 path and filter by pattern.

    Args:
        s3: S3 client to use.
        path: The S3 path to list from.
        allow_pattern: A list of patterns of which files to pull.
        ignore_pattern: A list of patterns of which files not to pull.

    Returns:
        tuple[str, str, list[str]]: A tuple where:
            - The first element is the bucket name
            - The second element is string represent the bucket
              and the prefix as a dir like string
            - The third element is a list of files allowed or
              disallowed by pattern
    """
    parts = path.removeprefix("s3://").split("/")
    prefix = "/".join(parts[1:])
    bucket_name = parts[0]

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj["Key"] for obj in objects.get("Contents", [])]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths


class S3Connector(BaseFileConnector):

    def __init__(self, url: str, **client_kwargs) -> None:
        import boto3
        from botocore.config import Config

        clean_url, parsed_kwargs = _parse_s3_kwargs(url)
        parsed_kwargs.update(client_kwargs)
        user_config = parsed_kwargs.pop("config", None)
        if user_config is not None and not isinstance(user_config, Config):
            raise TypeError(
                "S3Connector `config` must be a botocore.config.Config "
                f"instance, got {type(user_config).__name__}"
            )
        ua_extra = f"sglang/{_sglang_version}"
        if user_config is not None and user_config.user_agent_extra:
            ua_extra = f"{user_config.user_agent_extra} {ua_extra}"
        ua_overlay = Config(user_agent_extra=ua_extra)
        if user_config is not None:
            config = user_config.merge(ua_overlay)
        else:
            config = ua_overlay
        super().__init__(clean_url)
        # Drop ``None`` so opt-out sentinels don't reach boto3. A query/explicit
        # ``region_name`` is a direct kwarg here and outranks ``config.region_name``.
        self.client = boto3.client(
            "s3",
            config=config,
            **{k: v for k, v in parsed_kwargs.items() if v is not None},
        )

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        bucket_name, _, paths = list_files(
            self.client, path=self.url, allow_pattern=allow_pattern
        )
        return [f"s3://{bucket_name}/{path}" for path in paths]

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        """
        Pull files from S3 storage into the temporary directory.

        Args:
            s3_model_path: The S3 path of the model.
            allow_pattern: A list of patterns of which files to pull.
            ignore_pattern: A list of patterns of which files not to pull.

        """
        bucket_name, base_dir, files = list_files(
            self.client, self.url, allow_pattern, ignore_pattern
        )
        if len(files) == 0:
            return

        for file in files:
            destination_file = os.path.join(self.local_dir, file.removeprefix(base_dir))
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            self.client.download_file(bucket_name, file, destination_file)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        # only support safetensor files now
        hf_weights_files = self.glob(allow_pattern=["*.safetensors"])
        return runai_safetensors_weights_iterator(hf_weights_files)

    def close(self):
        # __init__ can fail during URL/kwarg validation (e.g. an empty query
        # value) before ``self.client`` or BaseConnector's state is set. Guard
        # the teardown so the destructor doesn't raise a noisy AttributeError.
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
        # BaseConnector.close() relies on state from BaseConnector.__init__
        # (self.closed / self.local_dir); skip it if that never ran.
        if getattr(self, "closed", None) is not None:
            super().close()
