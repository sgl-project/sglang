import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import RemoteInstanceModelLoader


@pytest.fixture(autouse=True)
def _runtime_server_args(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "get_server_args",
        lambda: SimpleNamespace(torchao_config=None),
    )


def test_heterogeneous_loader_builds_local_plan_and_reads_from_source(
    monkeypatch,
) -> None:
    calls = {}
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "fragments": [SimpleNamespace(fragment_id="target-fragment")],
    }

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeRegistrationLease:
        @classmethod
        def from_fragment(cls, fragment, *, runtime_lease_id=None):
            suffix = f":{runtime_lease_id}" if runtime_lease_id else ""
            return f"lease:{fragment.fragment_id}{suffix}"

    class FakeReader:
        def __init__(self, engine):
            calls["engine"] = engine

        def execute(self, plan, sources, target, **kwargs):
            assert calls["heartbeat_started"] is True
            calls["execute"] = (plan, sources, target, kwargs)
            return [SimpleNamespace(nbytes=64, operation_count=2)]

    class FakeHeartbeat:
        def __init__(self, seed_url, transfer_id, *, lease_timeout_sec):
            calls["heartbeat"] = (seed_url, transfer_id, lease_timeout_sec)

        def start(self):
            calls["heartbeat_started"] = True

        def raise_if_failed(self):
            calls["heartbeat_checked"] = calls.get("heartbeat_checked", 0) + 1

        def stop(self):
            calls["heartbeat_stopped"] = True

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FakeReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest

    def plan_runtime_transfer_to_local_target(sources, target):
        calls["plan"] = (sources, target)
        return "local-plan"

    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        plan_runtime_transfer_to_local_target
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: SimpleNamespace(
            transfer_id="transfer-1",
            manifests=[source_inventory],
            lease_timeout_sec=90,
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferHeartbeat",
        FakeHeartbeat,
    )
    monkeypatch.setattr(
        loader_module,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.setdefault(
            "released", (seed_url, transfer_id)
        ),
    )
    monkeypatch.setattr(
        loader_module.current_platform,
        "synchronize",
        lambda: calls.setdefault("synchronized", True),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: calls.setdefault("post_loaded", model),
    )

    @contextlib.contextmanager
    def target_builder(**kwargs):
        calls["builder"] = kwargs
        yield target_inventory

    model = object()
    engine = object()
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    success = loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        model,
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
    )

    assert success is True
    assert calls["builder"] == {
        "model": model,
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "instance_id": "sglang:target-session",
        "endpoint": "target-session",
    }
    assert calls["plan"][1].revision == source_inventory["revision"]
    _, _, _, execute_kwargs = calls["execute"]
    assert execute_kwargs == {
        "source_pre_registered": True,
        "source_registrations": ("lease:source-fragment:source-runtime-lease",),
        "target_pre_registered": True,
        "target_registrations": ("lease:target-fragment",),
    }
    assert calls["synchronized"] is True
    assert calls["post_loaded"] is model
    assert calls["heartbeat"] == ("http://seed:30000", "transfer-1", 90)
    assert calls["heartbeat_checked"] > 0
    assert calls["heartbeat_stopped"] is True
    assert calls["released"] == ("http://seed:30000", "transfer-1")


def test_post_load_weights_refreshes_gemma_runtime_buffer() -> None:
    norm = GemmaRMSNorm(4)
    norm.weight.data.copy_(torch.tensor([0.5, -0.25, 1.0, 2.0]))
    assert torch.equal(norm.gemma_weight, torch.ones(4))

    loader_module._post_load_weights(norm)

    assert torch.equal(norm.gemma_weight, norm.weight.data + 1.0)


def test_heterogeneous_loader_fails_closed_without_source_manifests(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        loader_module,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: None,
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )


def test_heterogeneous_loader_releases_source_snapshot_after_transfer_failure(
    monkeypatch,
) -> None:
    released = []
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [],
    }
    monkeypatch.setattr(
        loader_module,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: SimpleNamespace(
            transfer_id="transfer-1", manifests=[source_inventory]
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)),
    )

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")

    class FailingRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            raise RuntimeError("bad manifest")

    fake_weight_transfer.RuntimeManifest = FailingRuntimeManifest
    fake_weight_transfer.MemoryRegistrationLease = object
    fake_weight_transfer.MooncakeTransferEngineReader = object
    fake_weight_transfer.plan_runtime_transfer_to_local_target = object
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )
    assert released == [("http://seed:30000", "transfer-1")]


def test_heterogeneous_loader_fails_closed_when_heartbeat_fails_during_transfer(
    monkeypatch,
) -> None:
    state = {"released": [], "heartbeat_stopped": False}
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "fragments": [],
    }

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeHeartbeat:
        def __init__(self, seed_url, transfer_id, *, lease_timeout_sec):
            self.failed = False
            state["heartbeat"] = self

        def start(self):
            pass

        def raise_if_failed(self):
            if self.failed:
                raise RuntimeError("source lease renew failed")

        def stop(self):
            state["heartbeat_stopped"] = True

    class FakeReader:
        def __init__(self, engine):
            pass

        def execute(self, *args, **kwargs):
            state["heartbeat"].failed = True
            return [SimpleNamespace(nbytes=64, operation_count=1)]

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = SimpleNamespace(
        from_fragment=lambda fragment, **kwargs: fragment
    )
    fake_weight_transfer.MooncakeTransferEngineReader = FakeReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: object()
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: SimpleNamespace(
            transfer_id="transfer-1",
            manifests=[source_inventory],
            lease_timeout_sec=60,
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferHeartbeat",
        FakeHeartbeat,
    )
    monkeypatch.setattr(
        loader_module,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: state["released"].append((seed_url, transfer_id))
        or True,
    )
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(loader_module, "_post_load_weights", lambda model: None)

    @contextlib.contextmanager
    def target_builder(**kwargs):
        yield target_inventory

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            target_builder,
        )
        is False
    )
    assert state["heartbeat_stopped"] is True
    assert state["released"] == [("http://seed:30000", "transfer-1")]
