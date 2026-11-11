"""The NVFP4 quant info inherits the activation-scale mode the base layer's
quant config resolved, carries input scales in static mode only, and resolves
the per-token activation recipe from the flashinfer environment the stock
path reads, refusing any value the quantizers do not implement."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.lora.moe import quant_info as qi
from sglang.srt.lora.moe.runner import MoeLoraRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXPERTS, _HIDDEN, _INTER = 2, 256, 128


def _layer(*, per_token: bool, input_scale: float = 0.25) -> SimpleNamespace:
    """A ModelOpt NVFP4 FusedMoE as the quant info reads it, on CPU tensors."""

    def fp8(shape):
        return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e4m3fn)

    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        num_local_experts=_EXPERTS,
        intermediate_size_per_partition=_INTER,
        hidden_size=_HIDDEN,
        w13_weight=torch.zeros((_EXPERTS, 2 * _INTER, _HIDDEN // 2), dtype=torch.uint8),
        w2_weight=torch.zeros((_EXPERTS, _HIDDEN, _INTER // 2), dtype=torch.uint8),
        w13_weight_scale=fp8((_EXPERTS, 2 * _INTER, _HIDDEN // 16)),
        w2_weight_scale=fp8((_EXPERTS, _HIDDEN, _INTER // 16)),
        w13_weight_scale_2=torch.full((_EXPERTS,), 0.5),
        w2_weight_scale_2=torch.full((_EXPERTS,), 0.125),
        # The loader fills absent checkpoint scales with 1.0; a checkpoint
        # scale is any other value.
        w13_input_scale=torch.full((_EXPERTS, 2), input_scale),
        w2_input_scale=torch.full((_EXPERTS,), input_scale),
        quant_method=SimpleNamespace(
            quant_config=SimpleNamespace(use_per_token_activation=per_token)
        ),
    )


@pytest.mark.parametrize("input_scale", (0.25, 1.0))
def test_static_layer_reads_the_checkpoint_scales(input_scale):
    """Without the per-token flag the mode is static whatever the scales
    hold, the loader's 1.0 fill included: absence never selects per-token."""
    info = qi.MoeLoraNvFp4QuantInfo.from_layer(
        _layer(per_token=False, input_scale=input_scale)
    )
    assert info.act_scale_mode == "static"
    assert (info.w13_input_scale, info.w2_input_scale) == (input_scale, input_scale)
    assert info.recipe is None
    assert info.activation_denominator == 2688.0


@pytest.mark.parametrize("input_scale", (0.25, 1.0))
def test_per_token_layer_carries_no_input_scales(input_scale):
    """Under the flag the mode is per-token and the checkpoint scales are
    dropped, whatever their values, so nothing can fold them into the
    per-token path."""
    layer = _layer(per_token=True, input_scale=input_scale)
    assert qi.nvfp4_activation_scale_mode(layer) == "per_token"
    info = qi.MoeLoraNvFp4QuantInfo.from_layer(layer)
    assert info.act_scale_mode == "per_token"
    assert (info.w13_input_scale, info.w2_input_scale) == (None, None)
    assert info.recipe is None
    assert info.activation_denominator == 2688.0


def test_runner_mode_follows_nvfp4_layers_only():
    per_token = _layer(per_token=True)
    assert MoeLoraRunner._act_scale_mode(per_token, "nvfp4") == "per_token"
    assert MoeLoraRunner._act_scale_mode(_layer(per_token=False), "nvfp4") == "static"
    # Other families never see the flag, even on a layer that carries it.
    assert MoeLoraRunner._act_scale_mode(per_token, "mxfp4") == "static"
    assert MoeLoraRunner._act_scale_mode(per_token, "mxfp8") == "static"


def test_modes_and_scales_must_agree():
    kwargs = dict(
        w13_weight=torch.zeros((_EXPERTS, 2 * _INTER, _HIDDEN // 2), dtype=torch.uint8),
        w13_scale=torch.zeros((_EXPERTS, 2 * _INTER, _HIDDEN // 16), dtype=torch.uint8),
        w2_weight=torch.zeros((_EXPERTS, _HIDDEN, _INTER // 2), dtype=torch.uint8),
        w2_scale=torch.zeros((_EXPERTS, _HIDDEN, _INTER // 16), dtype=torch.uint8),
        elements_per_byte=2,
        num_local_experts=_EXPERTS,
        intermediate_size=_INTER,
        hidden_size=_HIDDEN,
        w13_global_scale=torch.ones(_EXPERTS),
        w2_global_scale=torch.ones(_EXPERTS),
    )
    recipe = qi.NvFp4ActivationRecipe(
        e4m3_max=448, err_mode="MAE", err_use_fast_math=False
    )
    with pytest.raises(ValueError, match="input scales"):
        qi.MoeLoraNvFp4QuantInfo(
            act_scale_mode="per_token",
            w13_input_scale=0.25,
            w2_input_scale=0.25,
            **kwargs,
        )
    with pytest.raises(ValueError, match="input scales"):
        qi.MoeLoraNvFp4QuantInfo(
            act_scale_mode="static", w13_input_scale=None, w2_input_scale=None, **kwargs
        )
    # One scale of the two is as wrong as both, in either mode.
    for w13_scale, w2_scale in ((0.25, None), (None, 0.25)):
        for mode in ("static", "per_token"):
            with pytest.raises(ValueError, match="input scales"):
                qi.MoeLoraNvFp4QuantInfo(
                    act_scale_mode=mode,
                    w13_input_scale=w13_scale,
                    w2_input_scale=w2_scale,
                    **kwargs,
                )
    with pytest.raises(ValueError, match="per-token"):
        qi.MoeLoraNvFp4QuantInfo(
            act_scale_mode="static",
            w13_input_scale=0.25,
            w2_input_scale=0.25,
            recipe=recipe,
            **kwargs,
        )
    info = qi.MoeLoraNvFp4QuantInfo(
        act_scale_mode="per_token",
        w13_input_scale=None,
        w2_input_scale=None,
        recipe=qi.NvFp4ActivationRecipe(
            e4m3_max=256, err_mode="MAE", err_use_fast_math=False
        ),
        **kwargs,
    )
    assert info.activation_denominator == 1536.0


def test_recipe_resolves_from_the_flashinfer_environment():
    assert qi.resolve_nvfp4_activation_recipe() is None
    with envs.FLASHINFER_NVFP4_4OVER6.override(True):
        assert qi.resolve_nvfp4_activation_recipe() == qi.NvFp4ActivationRecipe(
            e4m3_max=448, err_mode="MAE", err_use_fast_math=False
        )
        with (
            envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.override(True),
            envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override("mse"),
            envs.FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH.override(True),
        ):
            assert qi.resolve_nvfp4_activation_recipe() == qi.NvFp4ActivationRecipe(
                e4m3_max=256, err_mode="MSE", err_use_fast_math=True
            )
        with envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override("rms"):
            with pytest.raises(ValueError, match="MAE or MSE"):
                qi.resolve_nvfp4_activation_recipe()


@pytest.mark.parametrize(
    "use_256,err_mode,fast_math",
    list(itertools.product((False, True), ("MAE", "mse"), (False, True))),
)
def test_per_token_layer_carries_the_resolved_recipe(use_256, err_mode, fast_math):
    """Under the 4over6 environment a per-token layer resolves the recipe the
    stock path would (every value of the three fields), the normalization D
    follows e4m3_max (2688 for 448, 1536 for 256) while the input scales stay
    absent; a static layer ignores the environment entirely."""
    with (
        envs.FLASHINFER_NVFP4_4OVER6.override(True),
        envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.override(use_256),
        envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override(err_mode),
        envs.FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH.override(fast_math),
    ):
        info = qi.MoeLoraNvFp4QuantInfo.from_layer(_layer(per_token=True))
        assert info.act_scale_mode == "per_token"
        assert info.recipe == qi.NvFp4ActivationRecipe(
            e4m3_max=256 if use_256 else 448,
            err_mode=err_mode.upper(),
            err_use_fast_math=fast_math,
        )
        assert (info.w13_input_scale, info.w2_input_scale) == (None, None)
        assert info.activation_denominator == (1536.0 if use_256 else 2688.0)
        static = qi.MoeLoraNvFp4QuantInfo.from_layer(_layer(per_token=False))
        assert static.act_scale_mode == "static"
        assert static.recipe is None
        assert static.activation_denominator == 2688.0


def test_per_token_layer_refuses_an_unknown_recipe_field():
    """A field value outside the recipe the quantizers implement is refused at
    init and named: the error mode from the environment, and the two fields
    of a recipe built directly."""
    with (
        envs.FLASHINFER_NVFP4_4OVER6.override(True),
        envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override("rms"),
    ):
        with pytest.raises(ValueError, match="MAE or MSE, got 'RMS'"):
            qi.MoeLoraNvFp4QuantInfo.from_layer(_layer(per_token=True))
        # The static layer never resolves the recipe, so it never sees the error.
        assert (
            qi.MoeLoraNvFp4QuantInfo.from_layer(_layer(per_token=False)).recipe is None
        )
    with pytest.raises(ValueError, match="e4m3_max must be 256 or 448, got 512"):
        qi.NvFp4ActivationRecipe(e4m3_max=512, err_mode="MAE", err_use_fast_math=False)
    with pytest.raises(ValueError, match="err_mode must be MAE or MSE, got 'mae'"):
        qi.NvFp4ActivationRecipe(e4m3_max=448, err_mode="mae", err_use_fast_math=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
