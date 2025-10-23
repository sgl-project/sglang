"""Tests for --grpc-mode flag validation in RouterArgs."""

import pytest
from sglang_router.router_args import RouterArgs


def test_grpc_mode_default_false():
    """Test that grpc_mode defaults to False."""
    args = RouterArgs()
    assert args.grpc_mode is False


def test_grpc_mode_with_http_url_raises_error():
    """Test that --grpc-mode with http:// URL raises validation error."""
    args = RouterArgs(grpc_mode=True, worker_urls=["http://worker1:8000"])
    with pytest.raises(
        ValueError, match="--grpc-mode flag conflicts with HTTP/HTTPS worker URL"
    ):
        args._validate_grpc_mode()


def test_grpc_mode_with_https_url_raises_error():
    """Test that --grpc-mode with https:// URL raises validation error."""
    args = RouterArgs(grpc_mode=True, worker_urls=["https://worker1:8000"])
    with pytest.raises(
        ValueError, match="--grpc-mode flag conflicts with HTTP/HTTPS worker URL"
    ):
        args._validate_grpc_mode()


def test_grpc_mode_with_grpc_url_succeeds():
    """Test that --grpc-mode with grpc:// URL succeeds validation."""
    args = RouterArgs(grpc_mode=True, worker_urls=["grpc://worker1:8000"])
    # Should not raise
    args._validate_grpc_mode()


def test_grpc_mode_with_plain_url_succeeds():
    """Test that --grpc-mode with plain URL (no protocol) succeeds validation."""
    args = RouterArgs(grpc_mode=True, worker_urls=["worker1:8000", "worker2:8000"])
    # Should not raise
    args._validate_grpc_mode()


def test_grpc_mode_pd_prefill_http_url_raises_error():
    """Test that --grpc-mode with PD mode http:// prefill URL raises error."""
    args = RouterArgs(
        grpc_mode=True,
        pd_disaggregation=True,
        prefill_urls=[("http://prefill1:8000", None)],
        decode_urls=["grpc://decode1:8001"],
    )
    with pytest.raises(
        ValueError, match="--grpc-mode flag conflicts with HTTP/HTTPS prefill URL"
    ):
        args._validate_grpc_mode()


def test_grpc_mode_pd_decode_http_url_raises_error():
    """Test that --grpc-mode with PD mode http:// decode URL raises error."""
    args = RouterArgs(
        grpc_mode=True,
        pd_disaggregation=True,
        prefill_urls=[("grpc://prefill1:8000", None)],
        decode_urls=["http://decode1:8001"],
    )
    with pytest.raises(
        ValueError, match="--grpc-mode flag conflicts with HTTP/HTTPS decode URL"
    ):
        args._validate_grpc_mode()


def test_grpc_mode_without_flag_no_validation():
    """Test that http:// URLs are allowed when --grpc-mode is not set."""
    args = RouterArgs(grpc_mode=False, worker_urls=["http://worker1:8000"])
    # Should not raise - validation only applies when grpc_mode=True
    args._validate_grpc_mode()
