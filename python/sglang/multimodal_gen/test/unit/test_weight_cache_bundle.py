# SPDX-License-Identifier: Apache-2.0
"""Multi-component namespaces reuse the common manifest, mapping and budget."""

import gc
import os
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

from sglang.multimodal_gen.runtime.weight_cache.bundle import (
    build_bundle,
    component_storage_bytes,
    retain_importer,
    validate_manifest_components,
)
from sglang.multimodal_gen.runtime.weight_cache.client import (
    PROTOCOL,
    WeightCacheClient,
    validate_response,
)
from sglang.multimodal_gen.runtime.weight_cache.plan import CacheCompatibilityPlan
from sglang.multimodal_gen.test.unit.test_weight_cache_status import owner_fixture
from sglang.srt.weight_cache.common.liveness import ProcessIdentity
from sglang.srt.weight_cache.common.mapping import import_state, validate_meta_schema
from sglang.srt.weight_cache.common.transport import ExportGeneration
from sglang.srt.weight_cache.common.traversal import snapshot_module, storage_byte_views


def _components(device):
    first, second = torch.nn.Module(), torch.nn.Module()
    first.weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.float32, device=device).view(4, 4),
        requires_grad=False,
    )
    second.weight = first.weight
    second.register_buffer("slice", first.weight.detach()[::2, 1:], persistent=False)
    second.register_buffer("local", torch.full((4,), 3.0, device=device))
    first.eval()
    second.eval()
    return tuple(
        SimpleNamespace(
            name=name,
            load_ordinary=lambda m=model: (m, 0),
            build_meta=lambda m=model: m,
        )
        for name, model in (("dit", first), ("encoder", second))
    )


def test_whole_bundle_mapping_preserves_cross_component_ties_and_aliases():
    owner = build_bundle(_components("cpu"))
    snapshot = snapshot_module(owner)
    meta = build_bundle(_components("meta"), meta=True)
    validate_meta_schema(meta, snapshot.manifest)
    import_state(meta, snapshot.manifest, storage_byte_views(snapshot))
    assert meta["dit"].weight is meta["encoder"].weight
    assert (
        meta["encoder"].slice.untyped_storage().data_ptr()
        == meta["dit"].weight.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(meta["encoder"].slice, owner["encoder"].slice)
    assert "slice" in meta["encoder"]._non_persistent_buffers_set
    assert len(snapshot.manifest.storages) == 2
    assert snapshot.manifest.unique_storage_bytes == 80
    assert component_storage_bytes(snapshot.manifest, ("dit", "encoder")) == {
        "dit": 64,
        "encoder": 80,
    }


def test_component_lifetime_retains_the_guard_after_wrapper_is_dropped():
    bundle = build_bundle(_components("meta"), meta=True)
    importer = Mock()
    reference = weakref.ref(importer)
    retain_importer(bundle, importer)
    component = bundle["encoder"]
    del importer, bundle
    gc.collect()
    assert reference() is component._weight_cache_importer


@pytest.mark.parametrize(
    "names",
    [(), ("dit",), ("encoder",), ("dit", "unknown"), ("dit", "encoder", "encoder")],
)
def test_manifest_requires_the_entire_exact_component_set(names):
    manifest = snapshot_module(build_bundle(_components("cpu"))).manifest
    with pytest.raises(ValueError, match="component set"):
        validate_manifest_components(manifest, names)


def test_empty_and_duplicate_bundles_fail_closed():
    with pytest.raises(ValueError, match="at least one"):
        build_bundle(())
    components = _components("cpu")
    with pytest.raises(ValueError, match="distinct"):
        build_bundle((components[0], components[0]))


@pytest.mark.parametrize(
    "change",
    [
        {"protocol_version": 1},
        {"type": "fetch_component", "component": "transformer"},
        {"components": ["transformer"]},
        {"components": ["text_encoder", "transformer"]},
        {"components": ["transformer", "text_encoder", "text_encoder"]},
    ],
)
def test_incomplete_bundle_or_legacy_request_cannot_consume_budget(change):
    owner = owner_fixture()
    peer = ProcessIdentity.read(os.getpid())
    request = {
        **PROTOCOL,
        "type": "fetch_bundle",
        "compatibility": owner.plan.to_dict(),
        "components": ["transformer", "text_encoder"],
        "generation": msgspec.to_builtins(owner.exporter.generation),
        "consumer": msgspec.to_builtins(peer),
        "request_id": uuid.uuid4().hex,
        **change,
    }
    with pytest.raises(ValueError, match="protocol|bundle"):
        owner._request(request, peer)
    assert owner.exporter.stats()["deliveries_reserved"] == 0
    assert not owner.consumers
    owner.exporter._backend.export_entries.assert_not_called()


def test_client_rejects_legacy_owner_and_manifest_from_another_bundle():
    with pytest.raises(ValueError, match="protocol"):
        validate_response({**PROTOCOL, "protocol_version": 1, "status": "ok"})
    client = object.__new__(WeightCacheClient)
    client.plan = CacheCompatibilityPlan.from_fields(requested=["dit"])
    client.peer = ProcessIdentity.read(os.getpid())
    manifest = snapshot_module(build_bundle(_components("cpu"))).manifest
    generation = ExportGeneration(
        client.peer, "nonce", manifest.digest, "gpu", str(torch.__version__)
    )
    client.request = Mock(
        return_value={
            "compatibility": client.plan.to_dict(),
            "generation": msgspec.to_builtins(generation),
            "manifest": manifest.to_dict(),
        }
    )
    with pytest.raises(ValueError, match="component set"):
        client.manifest()
