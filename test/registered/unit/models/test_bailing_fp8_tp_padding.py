import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.bailing_fp8_tp_padding import (
    BailingFp8TPPadding,
    padded_bailing_fp8_intermediate_size,
)
from sglang.srt.runtime_context import get_flags, get_parallel, override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _quant_config(weight_block_size):
    return SimpleNamespace(
        get_name=lambda: "fp8",
        is_checkpoint_fp8_serialized=True,
        use_mxfp8=False,
        is_fp4_experts=False,
        weight_block_size=weight_block_size,
    )


def _owner(padded_size=256, hidden_size=128):
    return SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=torch.empty(
                (2 * padded_size, hidden_size), dtype=torch.float8_e4m3fn
            ),
            weight_scale_inv=torch.empty((2 * padded_size // 128, 1)),
        ),
        down_proj=SimpleNamespace(
            weight=torch.empty((hidden_size, padded_size), dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.empty((1, padded_size // 128)),
        ),
    )


def _dequantize(weight, scale):
    return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _block_scales(weight):
    blocks = weight.reshape(weight.shape[0] // 128, 128, weight.shape[1] // 128, 128)
    return blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0


def _load_shared_projection(helper, projection, weight, scale, *, load_scale=True):
    helper.load(projection, weight, is_scale=False)
    if load_scale:
        helper.load(projection, scale, is_scale=True)


def test_tp8_geometry_pads_after_logical_checkpoint_sharding():
    config = _quant_config([128, 128])

    assert padded_bailing_fp8_intermediate_size(config, 1280, 8) == 2048
    assert padded_bailing_fp8_intermediate_size(config, 1280, 4) == 1536
    assert padded_bailing_fp8_intermediate_size(config, 1280, 1) is None
    assert padded_bailing_fp8_intermediate_size(config, 1024, 8) is None
    assert (
        padded_bailing_fp8_intermediate_size(_quant_config([64, 128]), 1280, 8) is None
    )
    with pytest.raises(ValueError, match="divisible"):
        padded_bailing_fp8_intermediate_size(config, 1281, 8)


def test_hip_keeps_unpadded_tp_only_geometry():
    with override_platform(is_hip=True):
        assert (
            padded_bailing_fp8_intermediate_size(_quant_config([128, 128]), 1280, 8)
            is None
        )


_DEEP_GEMM = MoeRunnerBackend.DEEP_GEMM
_AUTO = MoeRunnerBackend.AUTO
_TRITON = MoeRunnerBackend.TRITON
_NONE = MoeA2ABackend.NONE
_DEEPEP = MoeA2ABackend.DEEPEP


_UE8M0_DEEP_GEMM_CASES = {
    # Ling-3.2 block FP8: EP keeps 1280; TP4/TP8 pad to 384/256 per rank.
    "ling32-ep-deep-gemm": (True, False, 1280, _DEEP_GEMM, _NONE, _DEEP_GEMM, True),
    "ling32-ep-auto-deepep": (True, False, 1280, _AUTO, _DEEPEP, _DEEP_GEMM, True),
    "ling32-ep-auto-a2a-none": (True, False, 1280, _AUTO, _NONE, _TRITON, False),
    "ling32-tp4-deep-gemm": (True, False, 384, _DEEP_GEMM, _NONE, _DEEP_GEMM, True),
    "ling32-tp8-deep-gemm": (True, False, 256, _DEEP_GEMM, _NONE, _DEEP_GEMM, True),
    # Hopper keeps the validated H200 DeepEP/DeepGEMM lane.
    "hopper-auto-deepep": (False, False, 1280, _AUTO, _DEEPEP, _DEEP_GEMM, False),
    # A 512-aligned expert width and MXFP8 are outside the observed failure.
    "aligned-deep-gemm": (True, False, 1024, _DEEP_GEMM, _NONE, _DEEP_GEMM, False),
    "aligned-auto-deepep": (True, False, 1024, _AUTO, _DEEPEP, _DEEP_GEMM, False),
    "mxfp8-deep-gemm": (True, True, 1280, _DEEP_GEMM, _NONE, _DEEP_GEMM, False),
    "mxfp8-auto-deepep": (True, True, 1280, _AUTO, _DEEPEP, _DEEP_GEMM, False),
}


@pytest.mark.parametrize(
    "ue8m0, use_mxfp8, intermediate_size_per_partition, runner, a2a, resolved, rejected",
    list(_UE8M0_DEEP_GEMM_CASES.values()),
    ids=list(_UE8M0_DEEP_GEMM_CASES),
)
def test_ue8m0_deep_gemm_rejection_covers_only_unaligned_block_fp8(
    ue8m0, use_mxfp8, intermediate_size_per_partition, runner, a2a, resolved, rejected
):
    """A resolved UE8M0 DeepGEMM runner is refused only for 128x128 block-FP8
    experts whose per-rank width is not a multiple of 512."""
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
    from sglang.srt.models.bailing_moe_v3 import _reject_unverified_ue8m0_deep_gemm

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        use_mxfp8=use_mxfp8,
        weight_block_size=None if use_mxfp8 else [128, 128],
    )
    with (
        patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", ue8m0),
        get_flags().moe.override(runner_backend=runner, a2a_backend=a2a),
    ):
        method = Fp8MoEMethod(quant_config)
        method.create_moe_runner(layer=None, moe_runner_config=MoeRunnerConfig())
        runner_backend = method.get_moe_runner_backend(layer=None)
        assert runner_backend == resolved

        expectation = (
            pytest.raises(NotImplementedError, match="--moe-runner-backend triton")
            if rejected
            else nullcontext()
        )
        with expectation:
            _reject_unverified_ue8m0_deep_gemm(
                quant_method=method,
                runner_backend=runner_backend,
                intermediate_size_per_partition=intermediate_size_per_partition,
            )


def test_shared_expert_reblock_uses_checkpoint_global_coordinates():
    torch.manual_seed(7)
    full_shape = (384, 128)
    full_scale_shape = (3, 1)
    weights = {
        projection: (torch.randn(full_shape) * 0.1).to(torch.float8_e4m3fn)
        for projection in ("gate_proj", "up_proj")
    }
    weights["down_proj"] = (torch.randn((128, 384)) * 0.1).to(torch.float8_e4m3fn)
    scales = {
        projection: torch.rand(full_scale_shape) + 0.5
        for projection in ("gate_proj", "up_proj")
    }
    scales["down_proj"] = torch.rand((1, 3)) + 0.5

    for rank in (0, 1):
        owner = _owner()
        helper = BailingFp8TPPadding(owner, 384, rank, 2, 128)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            _load_shared_projection(
                helper, projection, weights[projection], scales[projection]
            )
        helper.finalize()

        offset = rank * 192
        gate_expected = _dequantize(weights["gate_proj"], scales["gate_proj"])[
            offset : offset + 192
        ]
        up_expected = _dequantize(weights["up_proj"], scales["up_proj"])[
            offset : offset + 192
        ]
        down_expected = _dequantize(weights["down_proj"], scales["down_proj"])[
            :, offset : offset + 192
        ]
        gate_padded = F.pad(gate_expected, (0, 0, 0, 64))
        up_padded = F.pad(up_expected, (0, 0, 0, 64))
        down_padded = F.pad(down_expected, (0, 64))

        torch.testing.assert_close(
            owner.gate_up_proj.weight_scale_inv[:2],
            _block_scales(gate_padded),
        )
        torch.testing.assert_close(
            owner.gate_up_proj.weight_scale_inv[2:],
            _block_scales(up_padded),
        )
        torch.testing.assert_close(
            owner.down_proj.weight_scale_inv,
            _block_scales(down_padded),
        )
        torch.testing.assert_close(
            _dequantize(
                owner.gate_up_proj.weight[:256],
                owner.gate_up_proj.weight_scale_inv[:2],
            ),
            gate_padded,
            rtol=0.2,
            atol=0.1,
        )
        torch.testing.assert_close(
            _dequantize(
                owner.gate_up_proj.weight[256:],
                owner.gate_up_proj.weight_scale_inv[2:],
            ),
            up_padded,
            rtol=0.2,
            atol=0.1,
        )
        torch.testing.assert_close(
            _dequantize(owner.down_proj.weight, owner.down_proj.weight_scale_inv),
            down_padded,
            rtol=0.2,
            atol=0.1,
        )
        assert torch.count_nonzero(owner.gate_up_proj.weight[192:256].float()) == 0
        assert torch.count_nonzero(owner.gate_up_proj.weight[448:512].float()) == 0
        assert torch.count_nonzero(owner.down_proj.weight[:, 192:].float()) == 0


def test_finalize_rejects_a_missing_projection_scale():
    owner = _owner()
    helper = BailingFp8TPPadding(owner, 384, 0, 2, 128)
    for projection in ("gate_proj", "up_proj"):
        weight = torch.zeros((384, 128), dtype=torch.float8_e4m3fn)
        _load_shared_projection(
            helper,
            projection,
            weight,
            torch.ones((3, 1)),
            load_scale=projection != "up_proj",
        )
    _load_shared_projection(
        helper,
        "down_proj",
        torch.zeros((128, 384), dtype=torch.float8_e4m3fn),
        torch.ones((1, 3)),
    )

    with pytest.raises(ValueError, match="Missing.*scales"):
        helper.finalize()


def _parameter_owner(padded_size=256, hidden_size=128):
    def projection(shape, scale_shape):
        layer = nn.Module()
        layer.weight = nn.Parameter(
            (torch.randn(shape) * 0.1).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale_inv = nn.Parameter(
            torch.rand(scale_shape) + 0.5, requires_grad=False
        )
        return layer

    owner = nn.Module()
    owner.gate_up_proj = projection(
        (2 * padded_size, hidden_size), (2 * padded_size // 128, 1)
    )
    owner.down_proj = projection((hidden_size, padded_size), (1, padded_size // 128))
    return owner


def test_partial_online_update_is_rejected_before_mutating_reblocked_weights():
    """A partial online update must not overwrite reblocked FP8 weights and
    leave stale scales behind before the missing-tensor check rejects it."""
    torch.manual_seed(11)
    model = nn.Module()
    model.shared_experts = _parameter_owner()
    helper = BailingFp8TPPadding(model.shared_experts, 384, 0, 2, 128)

    def load_weights(weights):
        for name, value in weights:
            projection, suffix = name.split(".")[-2:]
            helper.load(projection, value, is_scale=suffix == "weight_scale_inv")
        helper.finalize()

    model.load_weights = load_weights
    before = {
        name: value.detach().view(torch.uint8).clone()
        for name, value in model.named_parameters()
    }
    partial = [
        (
            "shared_experts.gate_proj.weight",
            torch.ones((384, 128)).to(torch.float8_e4m3fn),
        ),
        ("shared_experts.gate_proj.weight_scale_inv", torch.ones((3, 1))),
    ]
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )
    updates = {
        "distributed": lambda: updater.load_weights_from_distributed(partial),
        "tensor": lambda: updater.update_weights_from_tensor(partial),
    }
    for source, update in updates.items():
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=0),
        ):
            success, message = update()
        assert not success, source
        assert "FP8 TP reblocking" in message, source
        for name, value in model.named_parameters():
            assert torch.equal(value.detach().view(torch.uint8), before[name]), (
                source,
                name,
            )
        assert helper.scales == {}, source
        assert helper.loaded_weights == set(), source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
