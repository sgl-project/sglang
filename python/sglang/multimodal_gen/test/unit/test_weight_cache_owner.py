# SPDX-License-Identifier: Apache-2.0
"""CPU control-plane regressions; no checkpoint/GPU model is needed."""

import os
import socket
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.runtime.weight_cache import daemon
from sglang.multimodal_gen.runtime.weight_cache.plan import CacheCompatibilityPlan
from sglang.multimodal_gen.test.unit.test_weight_cache_status import owner_fixture
from sglang.weight_cache_common.identity import default_runtime_dir


def test_device_and_socket_resources_both_exclude_second_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_WEIGHT_CACHE_DIR", str(tmp_path))
    first = owner_fixture()
    first.plan = CacheCompatibilityPlan.from_fields(rank={"device_uuid": "gpu-a"})
    first.path = tmp_path / "shared.sock"
    first.ready_path = first.path.with_suffix(".ready")
    second = owner_fixture()
    second.plan = CacheCompatibilityPlan.from_fields(rank={"device_uuid": "gpu-b"})
    second.path, second.ready_path = first.path, first.ready_path
    with ExitStack() as held:
        for path in first._owner_lock_paths():
            held.enter_context(daemon.owner_lock(path))
        with (
            pytest.raises(RuntimeError, match="already holds"),
            ExitStack() as contender,
        ):
            for path in second._owner_lock_paths():
                contender.enter_context(daemon.owner_lock(path))
        second.plan = first.plan
        second.path = tmp_path / "different.sock"
        second.ready_path = second.path.with_suffix(".ready")
        with (
            pytest.raises(RuntimeError, match="already holds"),
            ExitStack() as contender,
        ):
            for path in second._owner_lock_paths():
                contender.enter_context(daemon.owner_lock(path))


def test_stale_ready_recycled_pid_is_never_signalled(tmp_path):
    owner = owner_fixture()
    owner.path = tmp_path / "owner.sock"
    owner.ready_path = tmp_path / "owner.ready"
    owner.ready_path.write_text(f"pid={os.getpid()}\n")
    with socket.socket(socket.AF_UNIX) as stale:
        stale.bind(str(owner.path))
    with patch.object(daemon.os, "kill", side_effect=AssertionError("bare PID signal")):
        owner._cleanup_stale_files()
    assert not owner.path.exists() and not owner.ready_path.exists()


def test_live_socket_without_ready_cannot_be_stolen(tmp_path):
    owner = owner_fixture()
    owner.path, owner.ready_path = tmp_path / "owner.sock", tmp_path / "owner.ready"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(owner.path))
        listener.listen(1)
        with pytest.raises(RuntimeError, match="Live weight-cache socket"):
            owner._cleanup_stale_files()
        assert owner.path.exists()


def test_empty_and_relative_runtime_directories(monkeypatch):
    monkeypatch.delenv("SGLANG_DIFFUSION_WEIGHT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    assert default_runtime_dir().is_absolute()
    monkeypatch.setenv("SGLANG_DIFFUSION_WEIGHT_CACHE_DIR", "relative")
    with pytest.raises(ValueError, match="absolute"):
        default_runtime_dir()


def test_endpoint_cleanup_errors_do_not_skip_drain(caplog):
    owner = owner_fixture()
    owner.path = Mock()
    owner.ready_path = Mock()
    owner.ready_path.unlink.side_effect = PermissionError("denied")
    owner._remove_endpoints()
    owner.path.unlink.assert_called_once_with(missing_ok=True)
    assert "Cannot remove weight-cache endpoint" in caplog.text
