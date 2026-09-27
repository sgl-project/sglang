# SPDX-License-Identifier: Apache-2.0
"""Readiness probes must not race the health-success observer shutdown."""

import itertools
from unittest.mock import Mock, patch

import pytest
import requests

from sglang.multimodal_gen.test.single_test_file import test_weight_cache_1_gpu as wc


@pytest.mark.parametrize("observed", [True, False])
def test_health_success_preserves_or_probes_actual_liveness(observed):
    manager = wc.TimedServerManager.__new__(wc.TimedServerManager)
    manager.port = 12345
    manager.started = 10.0
    manager.readiness = {"liveness": 0.25} if observed else {}
    # Do not schedule the observer: deterministically make health win the race.
    with (
        patch.object(wc.threading, "Thread") as thread,
        patch.object(wc.ServerManager, "_wait_for_ready"),
        patch.object(wc.time, "perf_counter", side_effect=itertools.count(11.0)),
        patch.object(wc.requests, "get", return_value=Mock(status_code=200)) as get,
    ):
        manager._wait_for_ready(Mock(), Mock())
    assert manager.readiness["health"] == 1.0
    if observed:
        get.assert_not_called()
        assert manager.readiness["liveness"] == 0.25
    else:
        get.assert_called_once_with("http://127.0.0.1:12345/liveness", timeout=1)
        assert manager.readiness["liveness"] == 2.0


@pytest.mark.parametrize("timeout", [False, True])
def test_health_success_does_not_fabricate_liveness_on_probe_failure(timeout):
    manager = wc.TimedServerManager.__new__(wc.TimedServerManager)
    manager.port = 12345
    manager.started = 0.0
    manager.readiness = {}
    with (
        patch.object(wc.threading, "Thread") as thread,
        patch.object(wc.ServerManager, "_wait_for_ready"),
        patch.object(
            wc.requests,
            "get",
            side_effect=requests.Timeout("probe timeout") if timeout else None,
            return_value=Mock(status_code=503, text="not live"),
        ),
        pytest.raises(requests.Timeout if timeout else AssertionError),
    ):
        manager._wait_for_ready(Mock(), Mock())
    assert "liveness" not in manager.readiness
