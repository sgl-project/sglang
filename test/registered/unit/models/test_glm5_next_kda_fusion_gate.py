import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextLinearAttention,
    _fused_qkvbfg_is_unquantized,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# GLM-5.3-Flash names its KDA layers self_attn, and its checkpoint lists these
# projections under that prefix.
_PREFIX = "model.layers.0.self_attn"
# Every projection the two fused groups are built from.
_KDA_PROJECTIONS = [
    f"{_PREFIX}.{name}"
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
        "f_b_proj",
        "g_b_proj",
    )
]


def _fp8_config(skipped):
    return Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": skipped,
            "packed_modules_mapping": (
                Glm5NextForConditionalGeneration.packed_modules_mapping
            ),
        }
    )


def test_unquantized_checkpoint_fuses_without_the_env_gate():
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(False):
        assert _fused_qkvbfg_is_unquantized(quant_config=None, prefix=_PREFIX)


@pytest.mark.parametrize("gate", [False, True])
def test_quantized_kda_projections_never_fuse(gate):
    """Genuinely quantized projections must not fuse, gate set or not."""
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(gate):
        assert not _fused_qkvbfg_is_unquantized(
            quant_config=_fp8_config([]), prefix=_PREFIX
        )


@pytest.mark.parametrize(
    "quantized",
    [
        pytest.param(["f_a_proj"], id="qkvbfg-group"),
        pytest.param(["f_b_proj"], id="fg-b-group"),
    ],
)
def test_mixed_precision_group_declines_instead_of_raising(quantized):
    """A group whose projections disagree on precision falls back to the
    unfused path rather than failing to initialize."""
    skipped = [
        p for p in _KDA_PROJECTIONS if p.rsplit(".", maxsplit=1)[1] not in quantized
    ]
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True):
        assert not _fused_qkvbfg_is_unquantized(
            quant_config=_fp8_config(skipped), prefix=_PREFIX
        )


def test_fp8_checkpoint_that_skips_kda_fuses_only_when_gated():
    """An fp8 checkpoint that excludes the KDA projections fuses, but only
    when the gate is set."""
    config = _fp8_config(_KDA_PROJECTIONS)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True):
        assert _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(False):
        assert not _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)


def test_fused_projections_share_the_runtime_dtype():
    """Both fused projections follow the runtime dtype, not the checkpoint's;
    a mismatch raises 'expected scalar type Half but found BFloat16'."""
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "num_heads": 4,
            "head_dim": 16,
        },
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        set_default_torch_dtype(torch.float16),
    ):
        layer = Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=64,
            config=config,
            quant_config=None,
            prefix=_PREFIX,
        )
    assert layer.do_fuse_qkvbfg
    assert layer.fused_qkvbfg_a_proj.params_dtype == torch.float16
    assert layer.fused_fg_b_proj.weight.dtype == torch.float16


def test_eligible_layer_builds_unquantized():
    """An eligible layer builds its fused projection unquantized, whatever the
    quantizer would have resolved the fused name to on its own."""
    quant_config = BlockInt8Config.from_config(
        {
            "quant_method": "blockwise_int8",
            "activation_scheme": "dynamic",
            "ignored_layers": _KDA_PROJECTIONS,
            "weight_block_size": [128, 128],
        }
    )
    # Ask the quantizer the way construction does; checking is_layer_skipped
    # directly would keep passing if this config gained the packed mapping.
    probe = LinearBase(
        input_size=1,
        output_size=1,
        quant_config=quant_config,
        prefix=f"{_PREFIX}.fused_qkvbfg_a_proj",
    )
    assert not isinstance(probe.quant_method, UnquantizedLinearMethod), (
        "this quantizer must not resolve the fused name to an unquantized "
        "method, or the case is moot"
    )

    config = SimpleNamespace(
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "num_heads": 4,
            "head_dim": 16,
        },
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True),
        set_default_torch_dtype(torch.bfloat16),
    ):
        layer = Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=64,
            config=config,
            quant_config=quant_config,
            prefix=_PREFIX,
        )
    assert layer.do_fuse_qkvbfg
    assert isinstance(layer.fused_qkvbfg_a_proj.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
