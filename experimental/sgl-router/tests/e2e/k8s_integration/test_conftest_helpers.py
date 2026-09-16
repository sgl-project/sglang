import json
from types import SimpleNamespace

import conftest as k8s_conftest


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
