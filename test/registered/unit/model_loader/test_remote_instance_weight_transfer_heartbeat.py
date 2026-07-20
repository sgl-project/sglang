import importlib.util
import sys
import threading
from types import ModuleType

if importlib.util.find_spec("requests") is None:
    requests = ModuleType("requests")
    requests.post = None
    requests.delete = None
    sys.modules["requests"] = requests

from sglang.srt.model_loader import remote_instance_weight_loader_utils as utils


def test_begin_preserves_server_lease_timeout(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session.lease_timeout_sec == 90


def test_heartbeat_renews_in_background(monkeypatch) -> None:
    renewed = threading.Event()

    def renew(seed_url, transfer_id, lease_timeout_sec):
        renewed.set()
        return True

    monkeypatch.setattr(utils, "renew_remote_instance_weight_transfer", renew)
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    heartbeat.start()
    assert renewed.wait(timeout=1)
    heartbeat.raise_if_failed()
    heartbeat.stop()


def test_heartbeat_records_renew_failure_for_fail_closed_loader(monkeypatch) -> None:
    attempted = threading.Event()

    def renew(seed_url, transfer_id, lease_timeout_sec):
        attempted.set()
        return False

    monkeypatch.setattr(utils, "renew_remote_instance_weight_transfer", renew)
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    heartbeat.start()
    assert attempted.wait(timeout=1)

    try:
        heartbeat.raise_if_failed()
    except RuntimeError as error:
        assert "renew" in str(error).lower()
    else:
        raise AssertionError("heartbeat renewal failure must fail closed")
    finally:
        heartbeat.stop()
