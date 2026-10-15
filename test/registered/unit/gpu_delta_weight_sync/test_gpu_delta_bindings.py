"""Canonical format boundaries and prepared values, without serving dependencies."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.weight_sync.gpu_delta.bindings import (
    TensorBinding,
    _moe_binding,
    validate_ue8m0_payload,
)
from sglang.srt.weight_sync.gpu_delta.flashinfer import (
    DestinationSegment,
    FlashInferLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _block_scale_layer():
    return SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: (
            0 if expert == 2 else -1
        ),
        w2_weight_scale_inv=torch.full((1, 17, 15), -3.0),
    )


def test_block_scale_negotiation_is_identical_on_local_and_foreign_ranks():
    layout = FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    layer = _block_scale_layer()
    local, foreign = ({"dtype": "F32", "shape": [17, 15]} for _ in range(2))
    binding = _moe_binding("scale", local, layer, 2, "down", "weight_scale_inv", layout)
    assert (
        _moe_binding("scale", foreign, layer, 3, "down", "weight_scale_inv", layout)
        is None
    )
    expected = {
        "dtype": "U8",
        "shape": [17, 15],
        "checkpoint_dtype": "F32",
        "format": "fp8-block128-ue8m0",
        "encoding": "raw_bytes",
    }
    assert local == foreign == expected
    description = binding.describe()
    assert all(description[key] == value for key, value in expected.items())
    assert len(description["views"]) == 1


def test_ue8m0_expansion_is_exact_and_does_not_write_live_weights():
    layer = _block_scale_layer()
    original = layer.w2_weight_scale_inv.clone()
    binding = _moe_binding(
        "scale",
        {"dtype": "F32", "shape": [17, 15]},
        layer,
        2,
        "down",
        "weight_scale_inv",
        FlashInferLayout("fp8-block128-ue8m0", "trtllm", True),
    )
    payload = torch.arange(255, dtype=torch.uint8)
    validate_ue8m0_payload(payload)
    expanded = payload.view(torch.float8_e8m0fnu).to(torch.float32)
    ((target, source),) = binding.raw_sources(payload, expanded)
    expected_bits = torch.arange(255, dtype=torch.int32) << 23
    expected_bits[0] = 0x00400000
    assert torch.equal(source.view(torch.int32).flatten(), expected_bits)
    assert torch.equal(layer.w2_weight_scale_inv, original)
    assert target.data_ptr() == layer.w2_weight_scale_inv.data_ptr()
    assert source.data_ptr() == expanded.data_ptr()
    assert source.data_ptr() != target.data_ptr()
    with pytest.raises(ValueError, match="prepared FP32"):
        binding.raw_sources(payload)
    with pytest.raises(ValueError, match="NaN"):
        validate_ue8m0_payload(torch.full_like(payload, 255))
    assert torch.equal(layer.w2_weight_scale_inv, original)


def test_native_ue8m0_checkpoint_does_not_claim_fp32_normalization():
    binding = _moe_binding(
        "scale",
        {"dtype": "U8", "shape": [17, 15]},
        _block_scale_layer(),
        2,
        "down",
        "weight_scale_inv",
        FlashInferLayout("fp8-block128-ue8m0", "trtllm", True),
    )
    assert "checkpoint_dtype" not in binding.describe()
    with pytest.raises(ValueError, match="exact UE8M0"):
        _moe_binding(
            "scale",
            {"dtype": "I32", "shape": [17, 15]},
            _block_scale_layer(),
            2,
            "down",
            "weight_scale_inv",
            FlashInferLayout("fp8-block128-ue8m0", "trtllm", True),
        )


def test_apply_segments_keep_one_canonical_view_and_select_original_payload():
    target = torch.zeros((3, 5), dtype=torch.uint8)
    segments = (
        DestinationSegment(((0, 3), (0, 4)), (target[:, :4],)),
        DestinationSegment(((0, 3), (4, 5)), (target[:, 4:],)),
    )
    binding = TensorBinding(
        "weight", "U8", (3, 5), [[0, 3], [0, 5]], (target,), segments=segments
    )
    description = binding.describe()
    assert "format" not in description and "checkpoint_dtype" not in description
    payload = torch.arange(15, dtype=torch.uint8)
    proxies = binding.apply_bindings()
    assert binding.apply_bindings() is proxies
    assert len(proxies) == 2
    for proxy in proxies:
        assert not proxy.segments
        for destination in proxy.destinations:
            destination.bitwise_xor_(
                proxy.selected_bytes(payload).reshape(destination.shape)
            )
    assert torch.equal(target, payload.view(3, 5))
    assert binding.describe() == description
    assert len(description["views"]) == 1


def test_nvfp4_activation_calibration_is_global_but_weight_scales_are_local():
    layer = SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: (
            0 if expert == 2 else -1
        ),
        quant_config=SimpleNamespace(use_per_token_activation=False),
        w13_input_scale=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        w13_weight_scale_2=torch.ones(1, 2),
    )
    layout = FlashInferLayout("nvfp4", "trtllm", True, "w4a4")
    meta = {"dtype": "F32", "shape": []}
    binding = _moe_binding("activation", meta, layer, 3, "up", "input_scale", layout)
    assert binding.storage[0].data_ptr() == layer.w13_input_scale[3, 1].data_ptr()
    assert (
        _moe_binding("weight", meta, layer, 3, "up", "weight_scale_2", layout) is None
    )
    changed = torch.tensor([0.5]).view(torch.uint8)
    ((target, source),) = binding.raw_sources(changed)
    assert target.item() == 7.0 and source.item() == 0.5
    layer.quant_config.use_per_token_activation = True
    assert (
        _moe_binding("activation", meta, layer, 3, "up", "input_scale", layout) is None
    )


def test_multiple_scale_sources_share_one_expansion():
    shape = (2, 3)
    layout = FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    layer = SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: expert,
        w2_weight_scale_inv=torch.full((2, *shape), -3.0),
    )
    before = layer.w2_weight_scale_inv.clone()
    payload = torch.tensor([0, 1, 2, 127, 253, 254] * 2, dtype=torch.uint8)
    validate_ue8m0_payload(payload)
    expanded = payload.view(torch.float8_e8m0fnu).float()
    for expert in range(2):
        binding = _moe_binding(
            f"scale{expert}",
            {"dtype": "U8", "shape": list(shape)},
            layer,
            expert,
            "down",
            "weight_scale_inv",
            layout,
        )
        start = expert * 6
        ((target, source),) = binding.raw_sources(
            payload[start : start + 6], expanded[start : start + 6]
        )
        assert (
            source.untyped_storage().data_ptr() == expanded.untyped_storage().data_ptr()
        )
        assert source.storage_offset() == start
        assert target.data_ptr() == layer.w2_weight_scale_inv[expert].data_ptr()
    assert torch.equal(layer.w2_weight_scale_inv, before)
