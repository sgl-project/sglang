"""Packed MTP discovery and host-state transfer regressions without model weights."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs import model_config as model_config_module
from sglang.srt.configs.inkling import InklingMMConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
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


@pytest.mark.parametrize("local_drafts", [[0, 1, 2], [0, 2], []])
def test_inkling_checkpoint_packs_every_draft_pool(monkeypatch, tmp_path, local_drafts):
    # Checkpoints store the MTP depth outside the text config read by ModelConfig.
    config = InklingMMConfig(
        architectures=["InklingForConditionalGeneration"],
        num_nextn_predict_layers=3,
        mtp_config={"n_layers": 3, "local_layer_ids": local_drafts},
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
    # Each runner owns its draft depth while the checkpoint config is shared.
    for depth in range(3):
        layer_info = resolve_layer_indices(
            model=SimpleNamespace(mtp_layer_id_is_depth=True),
            model_config=draft_config,
            is_draft_worker=True,
            draft_model_idx=depth,
        )
        assert layer_info.num_effective_layers == 1
        assert layer_info.is_hybrid_swa_mtp_draft
        assert layer_info.swa_attention_layer_ids == (
            [depth] if depth in local_drafts else []
        )
        assert layer_info.full_attention_layer_ids == (
            [] if depth in local_drafts else [depth]
        )
    assert draft_config.swa_attention_layer_ids == local_drafts
    assert draft_config.full_attention_layer_ids == [
        i for i in range(3) if i not in local_drafts
    ]
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


@pytest.mark.parametrize("temporal_width", [0, 4])
def test_ascend_packed_state_roundtrip_uses_buffer_descriptors(
    monkeypatch, temporal_width
):
    monkeypatch.setenv("SGLANG_NPU_HICACHE_MAMBA_IO", "sync")
    infos = tuple(
        MambaStateBufferInfo(
            temporal=torch.arange(layers * 8 * temporal_width).reshape(
                layers, 8, temporal_width
            )
            + 100 * i,
            conv=(torch.arange(layers * 8 * 4).reshape(layers, 8, 4) + 100 * i,),
            extra_checkpoint_buffers=(torch.arange(16).reshape(8, 2) + 100 * i,),
        )
        for i, layers in enumerate((2, 1, 1, 1))
    )
    host = MambaPoolHost(
        infos[0],
        host_to_device_ratio=2,
        host_size=0,
        device_capacity=7,
        pin_memory=False,
        layout="page_first",
        mtp_draft_buffer_infos=infos[1:],
    )
    try:
        assert host.num_mamba_layers == 5
        assert host.temporal_device_ptrs.tolist() == [
            layer.data_ptr() for info in infos for layer in info.temporal.unbind(0)
        ]
        src_ids = torch.tensor([2, 5])
        host_ids = torch.tensor([1, 2])
        dst_ids = torch.tensor([3, 6])
        tensors = [
            layer
            for info in infos
            for tensor in (*info.conv, info.temporal)
            for layer in tensor.unbind(0)
        ] + [buf for info in infos for buf in info.extra_checkpoint_buffers]
        expected = [tensor[src_ids].clone() for tensor in tensors]
        host.backup_from_device_all_layer(None, host_ids, src_ids, "kernel_ascend")
        for tensor in tensors:
            tensor.zero_()
        for layer_id in range(host.num_mamba_layers):
            host.load_to_device_per_layer(
                None,
                host_ids,
                dst_ids,
                layer_id,
                "kernel_ascend",
                is_draft=layer_id >= 2,
            )
        for tensor, reference in zip(tensors, expected, strict=True):
            assert torch.equal(tensor[dst_ids], reference)
    finally:
        host.destroy()


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", "-x", *args]))
