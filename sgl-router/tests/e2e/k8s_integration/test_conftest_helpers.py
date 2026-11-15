import json
from types import SimpleNamespace

import conftest as k8s_conftest
import pytest


def _pod(name: str, phase: str, ready: bool) -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "phase": phase,
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True" if ready else "False",
                }
            ],
        },
    }


def test_wait_for_replacement_pod_ignores_old_and_pending_pods(monkeypatch):
    old_pod = "sgl-router-old"
    new_pod = "sgl-router-new"
    responses = iter(
        [
            [_pod(old_pod, "Running", True)],
            [
                _pod(old_pod, "Running", True),
                _pod(new_pod, "Running", True),
            ],
            [_pod(new_pod, "Pending", False)],
            [_pod(new_pod, "Running", True)],
        ]
    )
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps({"items": next(responses)}))

    monkeypatch.setattr(k8s_conftest, "_kubectl", fake_kubectl)
    monkeypatch.setattr(k8s_conftest.time, "sleep", lambda _: None)

    replacement = k8s_conftest._wait_for_replacement_pod_ready(
        old_pod,
        "app=sgl-router",
        timeout=5,
        interval=0,
    )

    assert replacement == new_pod
    assert len(calls) == 4
    assert all("-o" in args and "json" in args for args, _ in calls)


def _stub_kubectl(monkeypatch, payload):
    """Point every conftest helper at a canned kubectl response. `payload` is
    the object kubectl would have printed."""
    monkeypatch.setattr(
        k8s_conftest,
        "_kubectl",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )


def test_pod_names_excludes_terminating_pods(monkeypatch):
    terminating = _pod("sgl-router-old", "Running", True)
    terminating["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    _stub_kubectl(
        monkeypatch,
        {"items": [terminating, _pod("sgl-router-new", "Running", True)]},
    )

    assert k8s_conftest._pod_names("app=sgl-router") == ["sgl-router-new"]


def test_pods_returns_empty_on_a_failed_kubectl(monkeypatch):
    """The poll loops call this with `check=False` precisely because the API
    server can be briefly unavailable mid-rollout; a non-zero return must read
    as "nothing observed", not raise out of the loop."""
    monkeypatch.setattr(
        k8s_conftest,
        "_kubectl",
        lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=1),
    )

    assert k8s_conftest._pods("app=sgl-router", check=False) == []


def test_pods_tolerates_empty_stdout(monkeypatch):
    monkeypatch.setattr(
        k8s_conftest,
        "_kubectl",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )

    assert k8s_conftest._pods("app=sgl-router") == []


def test_container_restart_count_reads_the_named_container(monkeypatch):
    _stub_kubectl(
        monkeypatch,
        {
            "status": {
                "containerStatuses": [
                    {"name": "sidecar", "restartCount": 9},
                    {"name": "router", "restartCount": 3},
                ]
            }
        },
    )

    assert k8s_conftest._container_restart_count("sgl-router-0", "router") == 3


def test_container_restart_count_rejects_a_missing_container(monkeypatch):
    """`containerStatuses` lags during a restart, so the absent case is a real
    state — and the drain test reads its whole timing floor off this number.
    Returning 0 there would silently read as "never restarted"."""
    _stub_kubectl(monkeypatch, {"status": {"containerStatuses": []}})

    with pytest.raises(AssertionError, match="router"):
        k8s_conftest._container_restart_count("sgl-router-0", "router")


def test_pod_ready_condition_reports_the_ready_status(monkeypatch):
    _stub_kubectl(
        monkeypatch,
        {
            "status": {
                "conditions": [
                    {"type": "Initialized", "status": "True"},
                    {"type": "Ready", "status": "False"},
                ]
            }
        },
    )

    assert k8s_conftest._pod_ready_condition("sgl-router-0") == "False"


def test_pod_ready_condition_is_unknown_before_the_condition_exists(monkeypatch):
    """A pod whose Ready condition has not been written yet must read as
    "Unknown" rather than crash the drain test's diagnostic logging."""
    _stub_kubectl(monkeypatch, {"status": {"conditions": []}})

    assert k8s_conftest._pod_ready_condition("sgl-router-0") == "Unknown"
