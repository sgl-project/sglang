"""Packed MTP discovery and host-state transfer regressions without model weights."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs import model_config as model_config_module
from sglang.srt.configs.inkling import InklingMMConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.pool_host import mamba as mamba_module
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.speculative import base_spec_worker as spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _worker(monkeypatch, draft_config, target_pool, draft_pools):
    monkeypatch.setattr(
        spec, "get_memory", lambda: SimpleNamespace(enable_hierarchical_cache=True)
    )
    target = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        req_to_token_pool=SimpleNamespace(mamba_pool=target_pool),
    )
    runners = tuple(
        SimpleNamespace(
            model_config=draft_config,
            token_to_kv_pool=object(),
            req_to_token_pool=SimpleNamespace(mamba_pool=pool),
        )
        for pool in draft_pools
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target),
        _draft_model_runners=lambda: runners,
    )
    return worker, target, runners


def test_inkling_checkpoint_packs_every_draft_pool(monkeypatch, tmp_path):
    # Checkpoints store the MTP depth outside the text config read by ModelConfig.
    config = InklingMMConfig(
        architectures=["InklingForConditionalGeneration"],
        num_nextn_predict_layers=3,
        mtp_config={"n_layers": 3, "local_layer_ids": [0, 1, 2]},
        text_config={
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "local_layer_ids": [0, 2],
            "max_position_embeddings": 2048,
        },
    )
    monkeypatch.setattr(model_config_module, "get_config", lambda *a, **kw: config)
    monkeypatch.setattr(
        model_config_module, "get_generation_config", lambda *a, **kw: None
    )
    draft_config = ModelConfig(
        str(tmp_path), is_draft_model=True, is_multi_layer_eagle=True, dtype="bfloat16"
    )
    assert draft_config.num_nextn_predict_layers == 3
    assert draft_config.swa_attention_layer_ids == [0, 1, 2]
    assert draft_config.hf_config.architectures == [
        "InklingForConditionalGenerationMTP"
    ]
    worker, target, runners = _worker(
        monkeypatch, draft_config, object(), (object(), object(), object())
    )

    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)

    assert plan.mode == spec.HiCacheDraftMode.PACKED
    assert plan.device_pools == tuple(r.token_to_kv_pool for r in runners)
    assert target.mtp_draft_device_pools == plan.device_pools
    assert target.mtp_draft_mamba_pools == tuple(
        r.req_to_token_pool.mamba_pool for r in runners
    )


@pytest.mark.parametrize("has_state", [False, True])
def test_shared_target_state_is_not_packed_twice(monkeypatch, has_state):
    pool = object() if has_state else None
    worker, target, _ = _worker(
        monkeypatch, SimpleNamespace(num_nextn_predict_layers=3), pool, (pool,) * 3
    )
    target.mtp_draft_mamba_pools = (object(),)
    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode == spec.HiCacheDraftMode.PACKED
    assert target.mtp_draft_mamba_pools == ()


@pytest.mark.parametrize("missing_target", [False, True])
def test_separate_state_requires_compatible_slot_ownership(monkeypatch, missing_target):
    target_pool = None if missing_target else object()
    pools = (object(),) * 3 if missing_target else (object(), target_pool, object())
    worker, _, _ = _worker(
        monkeypatch, SimpleNamespace(num_nextn_predict_layers=3), target_pool, pools
    )
    with pytest.raises(AssertionError, match="target Mamba slot|separate state"):
        spec.BaseSpecWorker._build_hicache_draft_plan(worker)


@pytest.mark.parametrize(
    ("build_stack", "extra_kwargs"),
    [
        (assembler.build_hybrid_swa_stack, {"use_mla": False}),
        (
            assembler.build_hybrid_mamba_swa_stack,
            {
                "mamba_pool": object(),
                "mamba_layer_mapping": {},
                "page_size": 16,
                "tp_group": None,
            },
        ),
    ],
)
def test_swa_hicache_rejects_non_swa_drafts(build_stack, extra_kwargs):
    drafts = tuple(
        SimpleNamespace(
            full_kv_pool=SimpleNamespace(layer_num=int(i == 1)),
            swa_kv_pool=SimpleNamespace(layer_num=int(i != 1)),
        )
        for i in range(3)
    )
    with pytest.raises(AssertionError, match="requires SWA-only draft attention"):
        build_stack(
            params=SimpleNamespace(mtp_draft_device_pools=drafts),
            full_kv_pool=object(),
            swa_kv_pool=object(),
            full_layer_mapping={0: 0},
            swa_layer_mapping={1: 0},
            load_cache_event=None,
            storage_backend=None,
            **extra_kwargs,
        )


def test_ascend_packed_state_roundtrip_uses_per_layer_destinations(monkeypatch):
    # Exercise the portable NPU fallback on CPU. The bulk NPU transfer cannot
    # consume target and draft tensors from separate allocations.
    monkeypatch.setenv("SGLANG_NPU_HICACHE_MAMBA_IO", "sync")

    def unexpected_bulk_transfer(**kwargs):
        raise AssertionError("Packed state must not use the whole-pool NPU kernel")

    monkeypatch.setattr(mamba_module, "transfer_mamba_state", unexpected_bulk_transfer)
    layers = [torch.arange(32).reshape(8, 4) + 100 * i for i in range(5)]
    src_ids = torch.tensor([2, 5])
    host_ids = torch.tensor([1, 4])
    dst_ids = torch.tensor([3, 6])
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.layout = "page_first"
    host.mtp_draft_device_pools = (object(),) * 3
    host.num_mamba_layers = len(layers)
    host.temporal_state_elem_size = 0
    host.conv_state_shapes = [(4,)]
    host.conv_buffer = [torch.full((8, len(layers), 1, 4), -1)]
    MambaPoolHost._copy_tensor_all_layers_lf_pf(
        src_layers=layers,
        dst=host.conv_buffer[0],
        src_indices=src_ids,
        dst_indices=host_ids,
        num_layers=len(layers),
        io_backend="kernel_ascend",
        src_ptrs=torch.empty(0),
    )
    for i, source in enumerate(layers):
        assert torch.equal(host.conv_buffer[0][host_ids, i, 0], source[src_ids])
        is_draft = i >= 2
        destination = torch.full((1 if is_draft else 2, 8, 4), -1)
        pool = SimpleNamespace(mamba_cache=SimpleNamespace(conv=[destination]))
        host.load_to_device_per_layer(
            pool, host_ids, dst_ids, i, io_backend="kernel_ascend", is_draft=is_draft
        )
        expected = torch.full_like(destination, -1)
        expected[0 if is_draft else i, dst_ids] = source[src_ids]
        assert torch.equal(destination, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x", *sys.argv[1:]]))
