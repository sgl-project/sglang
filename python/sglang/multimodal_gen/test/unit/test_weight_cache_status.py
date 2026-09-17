# SPDX-License-Identifier: Apache-2.0

import os
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest

from sglang.multimodal_gen.runtime.weight_cache.client import (
    PROTOCOL,
    WeightCacheClient,
)
from sglang.multimodal_gen.runtime.weight_cache.daemon import DiffusionWeightCacheDaemon
from sglang.multimodal_gen.runtime.weight_cache.plan import CacheCompatibilityPlan
from sglang.weight_cache_common.liveness import ProcessIdentity
from sglang.weight_cache_common.transport import (
    CudaIpcExporter,
    ExportBudgetExceeded,
    ExportGeneration,
)


def owner_fixture():
    exporter = object.__new__(CudaIpcExporter)
    exporter._lock = threading.Lock()
    exporter._views = {0: Mock(), 1: Mock(), 2: Mock()}
    exporter._max_deliveries = 4
    exporter._max_storage_exports = 8
    exporter._requests = set()
    exporter._storage_exports = 0
    exporter._failed_deliveries = 0
    exporter._stopped = False
    exporter.manifest = Mock(unique_storage_bytes=100)
    exporter._backend = Mock()
    exporter._prepared = Mock()
    exporter.generation = ExportGeneration(
        ProcessIdentity.read(os.getpid()), "nonce", "digest", "gpu", "torch"
    )
    owner = object.__new__(DiffusionWeightCacheDaemon)
    owner._initialize_control()
    owner.plan = CacheCompatibilityPlan.from_fields(component={})
    owner.exporter = exporter
    return owner


def request(owner, kind, **fields):
    peer = ProcessIdentity.read(os.getpid())
    with patch(
        "sglang.multimodal_gen.runtime.weight_cache.daemon.ProcessHandle"
    ) as handle:
        handle.return_value.is_alive.return_value = True
        return owner._request(
            {
                **PROTOCOL,
                "type": kind,
                "compatibility": owner.plan.to_dict(),
                "consumer": msgspec.to_builtins(peer),
                **fields,
            },
            peer,
        )


def test_status_is_non_consuming_and_survives_nonrefundable_exhaustion():
    owner = owner_fixture()
    before = request(owner, "query_status")["cache_status"]
    assert before["storage_count"] == 3
    assert before["fetches_remaining"] == 2  # storage cap wins, not 4
    assert before["active_consumers"] == 0
    assert owner.consumers == {}
    owner.exporter._backend.export_entries.assert_not_called()
    # Failed deliveries reserve their entire budget; there is no refund.
    owner.exporter._backend.export_entries.side_effect = RuntimeError("partial export")
    for index in range(2):
        with pytest.raises(RuntimeError, match="partial export"):
            request(
                owner,
                "fetch_component",
                component="transformer",
                generation=msgspec.to_builtins(owner.exporter.generation),
                request_id=uuid.uuid4().hex,
            )
    after = request(owner, "query_status")["cache_status"]
    assert after["failed_deliveries"] == 2
    assert after["deliveries_reserved"] == 2
    assert after["storage_exports_reserved"] == 6
    assert after["deliveries_remaining"] == 2
    assert after["storage_exports_remaining"] == 2
    assert after["fetches_remaining"] == 0
    assert after["budget_exhausted"] and not after["accepting_fetches"]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        request(owner, "query_manifest")
    with pytest.raises(ExportBudgetExceeded):
        request(
            owner,
            "fetch_component",
            component="transformer",
            generation=msgspec.to_builtins(owner.exporter.generation),
            request_id=uuid.uuid4().hex,
        )
    # Dead consumers disappear from observations, but never refund sends.
    for handle in owner.consumers.values():
        handle.is_alive.return_value = False
    final = request(owner, "query_status")["cache_status"]
    assert final["active_consumers"] == 0
    assert final["storage_exports_reserved"] == 6


def test_status_validates_full_plan_and_authenticated_generation():
    owner = owner_fixture()
    client = WeightCacheClient.__new__(WeightCacheClient)
    client.plan = owner.plan
    client.peer = owner.exporter.generation.producer
    response = request(owner, "query_status")
    client.request = Mock(return_value=response)
    assert client.status()["fetches_remaining"] == 2
    client.request.assert_called_once_with("query_status")
    response["compatibility"] = {}
    with pytest.raises(ValueError, match="compatibility mismatch"):
        client.status()
    response["compatibility"] = owner.plan.to_dict()
    response["generation"]["producer"]["start_ticks"] += 1
    with pytest.raises(ValueError, match="authenticated socket peer"):
        client.status()


def test_stopped_exporter_is_observable_but_not_accepting():
    owner = owner_fixture()
    owner.exporter.stop_admission()
    status = request(owner, "query_status")["cache_status"]
    assert status["admission_stopped"] and not status["accepting_fetches"]
    assert status["fetches_remaining"] == 2


def test_status_cli_does_not_construct_owner_or_fetch():
    from sglang.multimodal_gen.runtime.weight_cache import daemon

    with (
        patch("sys.argv", ["daemon", "--status", "--model-path", "model"]),
        patch.object(
            daemon, "prepare_server_args", return_value=SimpleNamespace()
        ) as parse,
        patch.object(daemon, "resolve_pipeline_class"),
        patch.object(daemon, "prepare_pipeline"),
        patch.object(daemon, "compatibility_plan"),
        patch.object(daemon, "WeightCacheClient") as client,
        patch.object(daemon, "DiffusionWeightCacheDaemon") as owner,
    ):
        client.return_value.__enter__.return_value.status.return_value = {
            "fetches_remaining": 0
        }
        daemon.main()
        owner.assert_not_called()
        assert "--status" not in parse.call_args.args[0]
        client.return_value.__enter__.return_value.manifest.assert_not_called()
