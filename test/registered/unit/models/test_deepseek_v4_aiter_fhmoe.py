import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

pytestmark = pytest.mark.skipif(not is_hip(), reason="AITER FHMoE requires AMD ROCm")


def test_dsv4_topk_appends_semantic_shared_expert_384():
    hidden = torch.zeros(3, 16, device="cuda")
    logits = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    with patch(
        "sglang.srt.layers.moe.topk.get_parallel",
        return_value=SimpleNamespace(moe_ep_size=1),
    ):
        output = select_experts(
            hidden,
            logits,
            TopKConfig(
                top_k=7,
                correction_bias=torch.zeros(384, device="cuda", dtype=torch.float32),
                scoring_func="sqrtsoftplus",
                num_fused_shared_experts=1,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=True,
            ),
        )

    assert output.topk_ids.shape == (3, 7)
    assert torch.all(output.topk_ids[:, -1] == 384)
    assert torch.allclose(output.topk_weights[:, -1], torch.ones(3, device="cuda"))


def test_prepare_native_fp8_shared_expert_preserves_precision_and_padding(monkeypatch):
    monkeypatch.setattr(
        fp8,
        "get_parallel",
        lambda: SimpleNamespace(moe_tp_size=1, moe_tp_rank=0),
    )
    monkeypatch.setattr(fp8, "shuffle_weight", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(fp8, "shuffle_scale", lambda tensor, *args, **kwargs: tensor)

    gate = (
        torch.arange(128 * 128, dtype=torch.float32)
        .remainder(16)
        .reshape(128, 128)
        .to(torch.float8_e4m3fn)
    )
    up = (gate.float() + 1).to(torch.float8_e4m3fn)
    down = (gate.float() + 2).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 0x80, dtype=torch.uint8)
    layer = SimpleNamespace(
        use_aiter_fhmoe=True,
        w13_weight=torch.empty(385, 512, 64, dtype=torch.uint8),
        _fhmoe_native_weights={"w1": gate, "w2": down, "w3": up},
        _fhmoe_native_scales={
            "w1": scale.clone(),
            "w2": scale.clone(),
            "w3": scale.clone(),
        },
        fhmoe_shared_w1=None,
        fhmoe_shared_w2=None,
        fhmoe_shared_w1_scale=None,
        fhmoe_shared_w2_scale=None,
    )

    fp8._prepare_aiter_fhmoe_shared_expert(layer, 256, True)

    assert layer.fhmoe_shared_w1.dtype == torch.float8_e4m3fn
    assert layer.fhmoe_shared_w1.shape == (1, 512, 128)
    assert torch.equal(layer.fhmoe_shared_w1[0, :128], gate)
    assert torch.count_nonzero(layer.fhmoe_shared_w1[0, 128:256].float()) == 0
    assert torch.equal(layer.fhmoe_shared_w1[0, 256:384], up)
    assert torch.count_nonzero(layer.fhmoe_shared_w1[0, 384:].float()) == 0
    assert layer.fhmoe_shared_w2.shape == (1, 128, 256)
    assert torch.equal(layer.fhmoe_shared_w2[0, :, :128], down)
    assert torch.count_nonzero(layer.fhmoe_shared_w2[0, :, 128:].float()) == 0
    assert torch.all(layer.fhmoe_shared_w1_scale.view(torch.uint8)[128:256] == 0x7F)
    assert torch.all(layer.fhmoe_shared_w2_scale.view(torch.uint8)[:, 4:] == 0x7F)
    assert set(layer._fhmoe_native_weights) == {"w1", "w2", "w3"}
    assert set(layer._fhmoe_native_scales) == {"w1", "w2", "w3"}


def test_fused_moe_loader_keeps_only_native_fp8_tp_shard():
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer._has_fused_shared = True
    layer._num_local_routed = 384
    layer.quant_config = SimpleNamespace(is_fp4_experts=True)
    layer.use_aiter_fhmoe = True
    layer.moe_tp_size = 8
    layer.use_presharded_weights = False
    layer._pending_fp8_shared_weights = {}
    layer._pending_fp8_shared_scales = {}
    layer._fhmoe_load_lock = threading.Lock()
    layer._fhmoe_native_weights = {}
    layer._fhmoe_native_scales = {}
    layer.fhmoe_shared_w1 = None
    layer.fhmoe_shared_w2 = None
    layer.fhmoe_shared_w1_scale = None
    layer.fhmoe_shared_w2_scale = None
    layer.w13_weight = torch.nn.Parameter(torch.empty(1, 2))
    layer.w2_weight = torch.nn.Parameter(torch.empty(1))
    layer.w13_weight_scale_inv = torch.nn.Parameter(torch.empty(1))
    layer.w2_weight_scale_inv = torch.nn.Parameter(torch.empty(1))

    weight = torch.zeros((16, 8), dtype=torch.float8_e4m3fn)
    scale = torch.ones((8, 1), dtype=torch.uint8)
    assert layer._maybe_load_fp8_shared_expert_as_fp4(
        layer.w13_weight, weight, "weight", "w1", 384, 0, 3
    )
    assert layer._maybe_load_fp8_shared_expert_as_fp4(
        layer.w13_weight_scale_inv,
        scale,
        "weight_scale_inv",
        "w1",
        384,
        0,
        3,
    )

    assert layer._fhmoe_native_weights["w1"].shape == (2, 8)
    assert layer._fhmoe_native_scales["w1"].shape == (1, 1)
    assert layer._fhmoe_native_is_tp_sharded


def test_fhmoe_hot_reload_refreshes_derived_shared_tensors():
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer._has_fused_shared = True
    layer._num_local_routed = 384
    layer.quant_config = SimpleNamespace(is_fp4_experts=True)
    layer.use_aiter_fhmoe = True
    layer.moe_tp_size = 1
    layer.use_presharded_weights = True
    layer._pending_fp8_shared_weights = {}
    layer._pending_fp8_shared_scales = {}
    layer._fhmoe_load_lock = threading.Lock()
    old_weight = torch.zeros((2, 2), dtype=torch.float8_e4m3fn)
    old_scale = torch.ones((1, 1), dtype=torch.uint8)
    layer._fhmoe_native_weights = {"w2": old_weight, "w3": old_weight}
    layer._fhmoe_native_scales = {"w2": old_scale, "w3": old_scale}
    layer.fhmoe_shared_w1 = torch.empty(1)
    layer.fhmoe_shared_w2 = torch.empty(1)
    layer.fhmoe_shared_w1_scale = torch.empty(1)
    layer.fhmoe_shared_w2_scale = torch.empty(1)
    layer.w13_weight = torch.nn.Parameter(torch.empty(1, 2))
    layer.w2_weight = torch.nn.Parameter(torch.empty(1))
    layer.w13_weight_scale_inv = torch.nn.Parameter(torch.empty(1))
    layer.w2_weight_scale_inv = torch.nn.Parameter(torch.empty(1))

    new_weight = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    new_scale = torch.full((1, 1), 0x80, dtype=torch.uint8)
    with patch(
        "sglang.srt.layers.quantization.fp8._prepare_aiter_fhmoe_shared_expert"
    ) as prepare:
        layer._maybe_load_fp8_shared_expert_as_fp4(
            layer.w13_weight, new_weight, "weight", "w1", 384, 0, 0
        )
        layer._maybe_load_fp8_shared_expert_as_fp4(
            layer.w13_weight_scale_inv,
            new_scale,
            "weight_scale_inv",
            "w1",
            384,
            0,
            0,
        )

    prepare.assert_called_once_with(layer, 1, True)
    assert torch.equal(layer._fhmoe_native_weights["w1"], new_weight)
    assert torch.equal(layer._fhmoe_native_scales["w1"], new_scale)


def test_aiter_runner_forwards_heterogeneous_contract():
    captured = {}

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
    shared = (
        torch.empty(1, 512, 128, dtype=torch.float8_e4m3fn),
        torch.empty(1, 128, 256, dtype=torch.float8_e4m3fn),
        torch.empty(512, 8, dtype=torch.uint8),
        torch.empty(256, 8, dtype=torch.uint8),
    )
    quant_info = AiterMoeQuantInfo(
        w13_weight=torch.empty(385, 512, 64, dtype=torch.uint8),
        w2_weight=torch.empty(385, 128, 128, dtype=torch.uint8),
        quant_type=AiterQuantType.PER_1X32,
        shared_w1=shared[0],
        shared_w2=shared[1],
        shared_w1_scale=shared[2],
        shared_w2_scale=shared[3],
        shared_expert_id=384,
    )
    runner_input = AiterRunnerInput(
        hidden_states=hidden,
        topk_ids=torch.zeros(2, 7, dtype=torch.int32),
        topk_weights=torch.ones(2, 7, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )
    runner = AiterRunnerCore(
        SimpleNamespace(
            no_combine=False,
            activation="silu",
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        )
    )

    with (
        patch("aiter.fused_moe.fused_moe", side_effect=fake_fused_moe),
        patch(
            "sglang.srt.layers.moe.moe_runner.aiter."
            "aiter_fused_moe_supports_heterogeneous_shared_expert",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.moe.moe_runner.aiter._aiter_quant_type",
            return_value="per_1x32",
        ),
        patch(
            "sglang.srt.layers.moe.moe_runner.aiter._aiter_activation",
            return_value="silu",
        ),
    ):
        output = runner.run(runner_input, quant_info, {})

    assert output.hidden_states is hidden
    assert captured["shared_w1"] is shared[0]
    assert captured["shared_w2"] is shared[1]
    assert captured["shared_w1_scale"] is shared[2]
    assert captured["shared_w2_scale"] is shared[3]
    assert captured["shared_expert_id"] == 384
    assert captured["gate_mode"] == "interleave"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
