"""
Test suite for S3 CloudStorage integration.

Tests verify file upload, cleanup, URL generation, and error handling.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import sglang.multimodal_gen.runtime.entrypoints.openai.storage as storage_mod
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import CloudStorage


def _create_temp_file(tmp_path, name="test.png", content=b"\x89PNG\r\n\x1a\nfake"):
    """Create a temporary test file."""
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# UNIT TESTS


def test_upload_file_success(tmp_path):
    """Test successful upload with correct URL generation."""
    file_path = _create_temp_file(tmp_path, "image.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "my-bucket"
    storage_mod.cloud_storage.endpoint_url = "https://s3.example.com"
    storage_mod.cloud_storage.region_name = None

    called = {}

    def fake_upload(local_path, bucket, key, ExtraArgs=None):
        called["local_path"] = local_path
        called["bucket"] = bucket
        called["key"] = key
        called["extra"] = ExtraArgs

    storage_mod.cloud_storage.client = SimpleNamespace(upload_file=fake_upload)

    url = asyncio.run(storage_mod.cloud_storage.upload_file(file_path, "image.png"))

    assert url == "https://s3.example.com/my-bucket/image.png"
    assert called["local_path"] == file_path
    assert called["bucket"] == "my-bucket"
    assert called["key"] == "image.png"
    assert called["extra"]["ContentType"] == "image/png"


def test_upload_and_cleanup(tmp_path):
    """Test that local file is deleted after successful upload."""
    file_path = _create_temp_file(tmp_path, "cleanup.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "my-bucket"
    storage_mod.cloud_storage.endpoint_url = "https://s3.example.com"
    storage_mod.cloud_storage.client = SimpleNamespace(
        upload_file=lambda *args, **kwargs: None
    )

    assert os.path.exists(file_path)

    url = asyncio.run(storage_mod.cloud_storage.upload_and_cleanup(file_path))

    assert url == "https://s3.example.com/my-bucket/cleanup.png"
    assert not os.path.exists(file_path)


def test_upload_failure_preserves_file(tmp_path):
    """Test that file is preserved when upload fails."""
    file_path = _create_temp_file(tmp_path, "preserve.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "my-bucket"
    storage_mod.cloud_storage.endpoint_url = "https://s3.example.com"

    def fake_upload_raises(*args, **kwargs):
        raise RuntimeError("simulated failure")

    storage_mod.cloud_storage.client = SimpleNamespace(upload_file=fake_upload_raises)

    result = asyncio.run(storage_mod.cloud_storage.upload_and_cleanup(file_path))

    assert result is None
    assert os.path.exists(file_path)


def test_disabled_storage_returns_none(tmp_path):
    """Test that disabled storage returns None."""
    file_path = _create_temp_file(tmp_path, "test.png")

    prev_enabled = storage_mod.cloud_storage.enabled
    storage_mod.cloud_storage.enabled = False

    try:
        result = asyncio.run(
            storage_mod.cloud_storage.upload_file(file_path, "test.png")
        )
        assert result is None
    finally:
        storage_mod.cloud_storage.enabled = prev_enabled


def test_aws_url_with_region(tmp_path):
    """Test AWS S3 URL generation with specific region."""
    file_path = _create_temp_file(tmp_path, "aws.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "aws-bucket"
    storage_mod.cloud_storage.endpoint_url = None
    storage_mod.cloud_storage.region_name = "us-west-2"
    storage_mod.cloud_storage.client = SimpleNamespace(
        upload_file=lambda *args, **kwargs: None
    )

    url = asyncio.run(storage_mod.cloud_storage.upload_file(file_path, "aws.png"))

    assert url == "https://aws-bucket.s3.us-west-2.amazonaws.com/aws.png"


def test_aws_url_default_region(tmp_path):
    """Test AWS S3 URL defaults to us-east-1 when region not specified."""
    file_path = _create_temp_file(tmp_path, "default.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "default-bucket"
    storage_mod.cloud_storage.endpoint_url = None
    storage_mod.cloud_storage.region_name = None
    storage_mod.cloud_storage.client = SimpleNamespace(
        upload_file=lambda *args, **kwargs: None
    )

    url = asyncio.run(storage_mod.cloud_storage.upload_file(file_path, "default.png"))

    assert url == "https://default-bucket.s3.us-east-1.amazonaws.com/default.png"


def test_custom_endpoint_url(tmp_path):
    """Test URL generation with custom endpoint (MinIO/OSS/COS)."""
    file_path = _create_temp_file(tmp_path, "custom.png")

    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "custom-bucket"
    storage_mod.cloud_storage.endpoint_url = "https://minio.example.com/"
    storage_mod.cloud_storage.region_name = None
    storage_mod.cloud_storage.client = SimpleNamespace(
        upload_file=lambda *args, **kwargs: None
    )

    url = asyncio.run(storage_mod.cloud_storage.upload_file(file_path, "custom.png"))

    # Verify trailing slash is stripped
    assert url == "https://minio.example.com/custom-bucket/custom.png"


def test_content_type_detection(tmp_path):
    """Test Content-Type header for different file extensions."""
    storage_mod.cloud_storage.enabled = True
    storage_mod.cloud_storage.bucket_name = "test-bucket"
    storage_mod.cloud_storage.endpoint_url = "https://s3.test"

    test_cases = [
        ("image.png", "image/png"),
        ("image.jpg", "image/jpeg"),
        ("image.jpeg", "image/jpeg"),
        ("image.webp", "image/webp"),
        ("video.mp4", "video/mp4"),
        ("file.bin", "application/octet-stream"),
    ]

    for filename, expected_type in test_cases:
        called = {}

        def fake_upload(local_path, bucket, key, ExtraArgs=None):
            called["content_type"] = ExtraArgs.get("ContentType")

        storage_mod.cloud_storage.client = SimpleNamespace(upload_file=fake_upload)

        file_path = _create_temp_file(tmp_path, filename)
        asyncio.run(storage_mod.cloud_storage.upload_file(file_path, filename))

        assert called["content_type"] == expected_type


def test_env_configured_storage_uploads_and_cleans_up(tmp_path, monkeypatch):
    """CloudStorage() builds its boto3 client from the SGLANG_S3_* env vars."""
    monkeypatch.setenv("SGLANG_CLOUD_STORAGE_TYPE", "S3")
    monkeypatch.setenv("SGLANG_S3_BUCKET_NAME", "integration-test")
    monkeypatch.setenv("SGLANG_S3_REGION_NAME", "us-east-1")
    monkeypatch.setenv("SGLANG_S3_ACCESS_KEY_ID", "key-id")
    monkeypatch.setenv("SGLANG_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("SGLANG_S3_ENDPOINT_URL", raising=False)
    uploads = []
    client_kwargs = {}

    def fake_client(service, **kwargs):
        client_kwargs.update(kwargs, service=service)
        return SimpleNamespace(
            upload_file=lambda *args, **kw: uploads.append((args, kw))
        )

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))

    storage = CloudStorage()
    assert storage.is_enabled()
    assert client_kwargs == {
        "service": "s3",
        "aws_access_key_id": "key-id",
        "aws_secret_access_key": "secret",
        "endpoint_url": None,
        "region_name": "us-east-1",
    }

    file_path = _create_temp_file(tmp_path, "integration.png", b"test_data")
    url = asyncio.run(storage.upload_and_cleanup(file_path))

    assert url == (
        "https://integration-test.s3.us-east-1.amazonaws.com/integration.png"
    )
    assert not os.path.exists(file_path)
    assert uploads == [
        (
            (file_path, "integration-test", "integration.png"),
            {"ExtraArgs": {"ContentType": "image/png"}},
        )
    ]
