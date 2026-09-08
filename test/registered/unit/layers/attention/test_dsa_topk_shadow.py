import contextlib
import inspect
import json
import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import (
    logical_token_byte_offsets,
    restore_k_and_s_by_loc,
    snapshot_k_and_s_by_loc,
)
from sglang.srt.layers.attention.dsa import topk_shadow
from sglang.srt.layers.attention.dsa.topk_shadow import DSATopKShadowProbe
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _forward_batch(rid="probe-rid", carried=None, loc=None):
    if carried is None:
        carried = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    if loc is None:
        loc = torch.tensor([65], dtype=torch.int64)
    return SimpleNamespace(
        rids=[rid],
        out_cache_loc=loc,
        spec_info=SimpleNamespace(dsa_topk_indices=carried),
        _dsa_topk_shadow_callback=None,
        _dsa_topk_shadow_step=None,
    )


def test_logical_token_snapshot_restore_crosses_pages():
    page_size = 2
    index_head_dim = 4
    buf = torch.arange(3 * page_size * (index_head_dim + 4), dtype=torch.uint8).view(
        3, -1
    )
    loc = torch.tensor([0, 3, 4], dtype=torch.int64)
    offsets = logical_token_byte_offsets(
        buf=buf,
        loc=loc,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )
    snapshot = snapshot_k_and_s_by_loc(
        buf=buf,
        loc=loc,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )

    expected_offsets = torch.tensor(
        [
            [0, 1, 2, 3, 8, 9, 10, 11],
            [20, 21, 22, 23, 28, 29, 30, 31],
            [32, 33, 34, 35, 40, 41, 42, 43],
        ],
        dtype=torch.int64,
    )
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(snapshot, buf.flatten()[expected_offsets])

    original = buf.clone()
    buf.flatten()[offsets] = 0
    restore_k_and_s_by_loc(
        buf=buf,
        loc=loc,
        snapshot=snapshot,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )
    assert torch.equal(buf, original)


def test_logical_token_offsets_match_preshuffled_layout():
    buf = torch.zeros((1, 2 * (4 + 4)), dtype=torch.uint8)
    offsets = logical_token_byte_offsets(
        buf=buf,
        loc=torch.tensor([0, 1]),
        page_size=2,
        index_head_dim=4,
        preshuffle_tile=2,
    )
    assert torch.equal(
        offsets,
        torch.tensor(
            [
                [0, 1, 4, 5, 8, 9, 10, 11],
                [2, 3, 6, 7, 12, 13, 14, 15],
            ],
            dtype=torch.int64,
        ),
    )


def test_disabled_probe_is_noop():
    probe = DSATopKShadowProbe(None)
    batch = _forward_batch()

    with probe.forward_scope(batch, step=0, using_cuda_graph=False):
        assert batch._dsa_topk_shadow_callback is None

    assert not probe._seen
    assert not probe._buckets


@pytest.mark.parametrize(
    "is_nextn,has_callback,has_carried,expected_calls",
    [
        (True, True, True, 1),
        (False, True, True, 0),
        (True, False, True, 0),
        (True, True, False, 0),
    ],
)
def test_mla_shadow_callback_requires_nextn_carry_and_explicit_arm(
    is_nextn, has_callback, has_carried, expected_calls
):
    calls = []
    callback = (lambda **kwargs: calls.append(kwargs)) if has_callback else None
    attention = SimpleNamespace(
        is_nextn=is_nextn,
        indexer=object(),
        layer_id=7,
    )
    batch = SimpleNamespace(_dsa_topk_shadow_callback=callback)
    hidden_states = torch.zeros((1, 4))
    q_lora = torch.ones((1, 2))
    positions = torch.tensor([3])
    carried = torch.tensor([[1, 2]]) if has_carried else None

    DeepseekMLAForwardMixin._maybe_compare_carried_dsa_topk(
        attention,
        hidden_states=hidden_states,
        q_lora=q_lora,
        positions=positions,
        forward_batch=batch,
        prev_topk_indices=carried,
    )

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls[0] == {
            "indexer": attention.indexer,
            "x": hidden_states,
            "q_lora": q_lora,
            "positions": positions,
            "forward_batch": batch,
            "layer_id": 7,
        }


def test_probe_tracks_seed_then_step_zero_publish_in_order(monkeypatch):
    probe = DSATopKShadowProbe("probe-rid")
    batch = _forward_batch(carried=torch.tensor([[10, 11]], dtype=torch.int32))
    calls = []

    def compare(**kwargs):
        calls.append((kwargs["step"], kwargs["carried"].clone()))

    monkeypatch.setattr(probe, "compare", compare)
    for step, carried in (
        (0, torch.tensor([[10, 11]], dtype=torch.int32)),
        (1, torch.tensor([[20, 21]], dtype=torch.int32)),
    ):
        batch.spec_info.dsa_topk_indices = carried
        with probe.forward_scope(batch, step=step, using_cuda_graph=False):
            batch._dsa_topk_shadow_callback(indexer=None)

    assert [step for step, _ in calls] == [0, 1]
    assert torch.equal(calls[0][1], torch.tensor([[10, 11]], dtype=torch.int32))
    assert torch.equal(calls[1][1], torch.tensor([[20, 21]], dtype=torch.int32))
    assert batch._dsa_topk_shadow_callback is None
    assert batch._dsa_topk_shadow_step is None


@pytest.mark.parametrize(
    "batch,using_cuda_graph,reason",
    [
        (
            _forward_batch(rid="probe-rid"),
            True,
            "CUDA graph execution is unsupported",
        ),
        (
            SimpleNamespace(
                rids=["probe-rid", "other"],
                spec_info=SimpleNamespace(
                    dsa_topk_indices=torch.tensor([[1], [2]], dtype=torch.int32)
                ),
                _dsa_topk_shadow_callback=None,
                _dsa_topk_shadow_step=None,
            ),
            False,
            "target rid must be the sole request in the batch",
        ),
    ],
)
def test_probe_fails_closed_for_graph_or_multi_request(batch, using_cuda_graph, reason):
    probe = DSATopKShadowProbe("probe-rid")
    with probe.forward_scope(batch, step=0, using_cuda_graph=using_cuda_graph):
        assert batch._dsa_topk_shadow_callback is None
    assert probe._rejection == reason


@pytest.mark.parametrize(
    "runtime,reason",
    [
        (
            {"cuda": False, "seed": True, "fused": False},
            "requires CUDA",
        ),
        (
            {"cuda": True, "seed": False, "fused": False},
            "requires draft-extend DSA seed",
        ),
        (
            {"cuda": True, "seed": True, "fused": True},
            "requires request-relative unfused DSA TopK",
        ),
    ],
)
def test_runtime_contract_fails_closed(monkeypatch, runtime, reason):
    monkeypatch.setattr(topk_shadow, "is_cuda", lambda: runtime["cuda"])
    monkeypatch.setattr(
        topk_shadow,
        "should_use_dsa_fused_topk",
        lambda seed_enabled: runtime["fused"],
    )
    monkeypatch.setattr(
        topk_shadow,
        "get_parallel",
        lambda: SimpleNamespace(
            attn_cp_size=1, dcp_enabled=False, enable_dsa_cache_layer_split=False
        ),
    )
    monkeypatch.setattr(
        topk_shadow,
        "get_memory",
        lambda: SimpleNamespace(enable_hisparse=False),
    )

    probe = DSATopKShadowProbe.from_runtime("probe-rid", seed_enabled=runtime["seed"])

    assert not probe.can_probe
    assert probe._rejection == reason


@pytest.mark.parametrize(
    "parallel_updates,memory_updates,reason",
    [
        ({"attn_cp_size": 2}, {}, "attn_cp_size must be 1"),
        ({"dcp_enabled": True}, {}, "DCP is unsupported"),
        (
            {"enable_dsa_cache_layer_split": True},
            {},
            "DSA cache layer split is unsupported",
        ),
        ({}, {"enable_hisparse": True}, "HiSparse is unsupported"),
    ],
)
def test_runtime_topology_contract_fails_closed(
    monkeypatch, parallel_updates, memory_updates, reason
):
    parallel = {
        "attn_cp_size": 1,
        "dcp_enabled": False,
        "enable_dsa_cache_layer_split": False,
    }
    memory = {"enable_hisparse": False}
    parallel.update(parallel_updates)
    memory.update(memory_updates)
    monkeypatch.setattr(topk_shadow, "is_cuda", lambda: True)
    monkeypatch.setattr(
        topk_shadow, "should_use_dsa_fused_topk", lambda seed_enabled: False
    )
    monkeypatch.setattr(
        topk_shadow, "get_parallel", lambda: SimpleNamespace(**parallel)
    )
    monkeypatch.setattr(topk_shadow, "get_memory", lambda: SimpleNamespace(**memory))

    probe = DSATopKShadowProbe.from_runtime("probe-rid", seed_enabled=True)

    assert not probe.can_probe
    assert probe._rejection == reason


def test_finish_without_samples_is_rejected(caplog):
    probe = DSATopKShadowProbe("probe-rid")
    with caplog.at_level(logging.WARNING):
        probe.finish(rid="probe-rid", natural_stop=True)
    line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("DSA_TOPK_SHADOW_RESULT ")
    )
    result = json.loads(line.removeprefix("DSA_TOPK_SHADOW_RESULT "))
    assert result["status"] == "rejected"
    assert result["rejection"] == "target request produced no complete shadow samples"


def test_shadow_restores_index_k_and_suppresses_ordinary_capture(monkeypatch, caplog):
    page_size = 64
    index_head_dim = 128
    buf = (
        torch.arange(2 * page_size * (index_head_dim + 4), dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(2, -1)
    )
    original = buf.clone()
    pool = SimpleNamespace(
        page_size=page_size,
        index_head_dim=index_head_dim,
        get_index_k_with_scale_buffer=lambda layer_id: buf,
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
        lambda: pool,
    )
    capture_suspended = False

    @contextlib.contextmanager
    def suspend_capture():
        nonlocal capture_suspended
        capture_suspended = True
        try:
            yield
        finally:
            capture_suspended = False

    monkeypatch.setattr(topk_shadow, "suspend_indexer_topk_capture", suspend_capture)
    batch = _forward_batch()
    offsets = logical_token_byte_offsets(
        buf=buf,
        loc=batch.out_cache_loc,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )

    def indexer(**kwargs):
        assert capture_suspended
        buf.flatten()[offsets] = 255
        return torch.tensor([[1, 2, -1]], dtype=torch.int32)

    probe = DSATopKShadowProbe("probe-rid")
    monkeypatch.setattr(topk_shadow, "get_is_capture_mode", lambda: False)
    probe.compare(
        step=0,
        carried=batch.spec_info.dsa_topk_indices,
        indexer=indexer,
        x=torch.zeros((1, 4)),
        q_lora=torch.zeros((1, 4)),
        positions=torch.zeros((1,), dtype=torch.int64),
        forward_batch=batch,
        layer_id=7,
    )

    assert torch.equal(buf, original)
    metrics = next(iter(probe._buckets.values())).tolist()
    assert metrics == [0, 1, 0, 2, 2, 1, 5, 8]

    with caplog.at_level(logging.WARNING):
        probe.finish(rid="probe-rid", natural_stop=True)
    line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("DSA_TOPK_SHADOW_RESULT ")
    )
    result = json.loads(line.removeprefix("DSA_TOPK_SHADOW_RESULT "))
    assert result["status"] == "complete"
    assert result["buckets"][0]["provenance"] == "draft_extend_seed"
    assert result["buckets"][0]["valid_set_intersection"] == 1


def test_shadow_restores_index_k_when_indexer_raises(monkeypatch):
    page_size = 64
    index_head_dim = 128
    buf = (
        torch.arange(2 * page_size * (index_head_dim + 4), dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(2, -1)
    )
    original = buf.clone()
    pool = SimpleNamespace(
        page_size=page_size,
        index_head_dim=index_head_dim,
        get_index_k_with_scale_buffer=lambda layer_id: buf,
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
        lambda: pool,
    )
    monkeypatch.setattr(topk_shadow, "get_is_capture_mode", lambda: False)
    batch = _forward_batch()
    offsets = logical_token_byte_offsets(
        buf=buf,
        loc=batch.out_cache_loc,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )

    def indexer(**kwargs):
        buf.flatten()[offsets] = 255
        raise RuntimeError("shadow failed")

    probe = DSATopKShadowProbe("probe-rid")
    with pytest.raises(RuntimeError, match="shadow failed"):
        probe.compare(
            step=0,
            carried=batch.spec_info.dsa_topk_indices,
            indexer=indexer,
            x=torch.zeros((1, 4)),
            q_lora=torch.zeros((1, 4)),
            positions=torch.zeros((1,), dtype=torch.int64),
            forward_batch=batch,
            layer_id=7,
        )

    assert torch.equal(buf, original)


def test_shadow_accepts_quantized_tuple_input(monkeypatch):
    page_size = 64
    index_head_dim = 128
    buf = torch.zeros((2, page_size * (index_head_dim + 4)), dtype=torch.uint8)
    pool = SimpleNamespace(
        page_size=page_size,
        index_head_dim=index_head_dim,
        get_index_k_with_scale_buffer=lambda layer_id: buf,
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
        lambda: pool,
    )
    monkeypatch.setattr(topk_shadow, "get_is_capture_mode", lambda: False)
    batch = _forward_batch()
    probe = DSATopKShadowProbe("probe-rid")

    probe.compare(
        step=0,
        carried=batch.spec_info.dsa_topk_indices,
        indexer=lambda **kwargs: torch.tensor([[0, 1, -1]], dtype=torch.int32),
        x=(torch.zeros((1, 4)), torch.ones((1, 1))),
        q_lora=torch.zeros((1, 4)),
        positions=torch.zeros((1,), dtype=torch.int64),
        forward_batch=batch,
        layer_id=7,
    )

    assert probe._rejection is None
    assert next(iter(probe._buckets.values()))[0].item() == 1


def test_hot_path_has_no_scalar_or_host_materialization():
    source = inspect.getsource(DSATopKShadowProbe.compare) + inspect.getsource(
        DSATopKShadowProbe._accumulate
    )
    assert ".item(" not in source
    assert ".tolist(" not in source
    assert ".cpu(" not in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
