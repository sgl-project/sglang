# SPDX-License-Identifier: Apache-2.0
"""The aiter_quant format table, checked against aiter's own contract
(ROCm-only; skipped elsewhere).

Rows are (Q/K, V) pairs plus a scale recipe that `aiter.ops.mha_v4` validates
at call time, on the GPU, behind a ROCm-only import -- so a row aiter rejects
(MXFP4 Q/K with FP8 V was one) otherwise surfaces as a runtime failure in a
generation run rather than here.
"""

import enum
import types
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import aiter_quant
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="aiter_quant is a ROCm backend",
)

HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5

_FORMAT_NAMES = [fmt.name for fmt in aiter_quant._FORMATS]

# aiter's canonical MXFP8 recipe: FP8 operands, E8M0 1x32 Q/K scales, per-tensor
# V scale. Copied from aiter, which rejects any other triple for FP8 formats.
_MXFP8_SCALE_MODE_NAMES = (
    "E8M0_PER_1X32",
    "E8M0_PER_1X32",
    "F32_PER_TENSOR",
)


def _aiter_mha_v4():
    # `aiter` ships with ROCm only, and mha_v4 needs a recent build.
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    return pytest.importorskip(
        "aiter.ops.mha_v4", reason="this aiter build has no mha_v4"
    )


def test_default_format_is_fp8():
    assert aiter_quant._DEFAULT_FORMAT == "fp8"
    assert aiter_quant._DEFAULT_FORMAT in aiter_quant._FORMATS_BY_NAME


def test_unknown_format_names_the_supported_ones():
    with pytest.raises(ValueError, match="Unknown aiter_quant format 'f4f4'"):
        aiter_quant._lookup_format("f4f4")


def test_gfx942_admits_only_the_fp8_and_int8_rows():
    # The gfx942 fmha_v4 manifest carries no BF16 Q/K row and no MX row, so
    # selecting one there must raise rather than reach the dispatcher and miss.
    with (
        patch.object(aiter_quant, "is_gfx95_supported", return_value=False),
        patch.object(aiter_quant, "is_gfx942_supported", return_value=True),
    ):
        for fmt in aiter_quant._FORMATS:
            if fmt.gfx942:
                aiter_quant._validate_arch(fmt)
            else:
                with pytest.raises(NotImplementedError, match="no gfx942 kernel row"):
                    aiter_quant._validate_arch(fmt)


def test_neither_arch_raises():
    with (
        patch.object(aiter_quant, "is_gfx95_supported", return_value=False),
        patch.object(aiter_quant, "is_gfx942_supported", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="gfx950- or gfx942-class"):
            aiter_quant._validate_arch(aiter_quant._FORMATS[0])


def test_mxfp8_is_the_only_block_scaled_row_and_uses_aiter_recipe():
    block_scaled = [f.name for f in aiter_quant._FORMATS if f.block_scaled]
    assert block_scaled == ["mxfp8"]

    scale_mode = enum.Enum("AttentionScaleMode", sorted(set(_MXFP8_SCALE_MODE_NAMES)))
    with patch.object(aiter_quant, "_AiterAttentionScaleMode", scale_mode):
        kwargs = aiter_quant._scale_mode_kwargs(aiter_quant._FORMATS_BY_NAME["mxfp8"])
    assert tuple(mode.name for mode in kwargs.values()) == _MXFP8_SCALE_MODE_NAMES
    assert list(kwargs) == ["q_scale_mode", "k_scale_mode", "v_scale_mode"]


def test_non_block_scaled_rows_pass_no_scale_modes():
    # aiter derives the canonical modes from the format pair and rejects a
    # non-canonical triple, so these rows must leave them unset.
    for fmt in aiter_quant._FORMATS:
        if not fmt.block_scaled:
            assert aiter_quant._scale_mode_kwargs(fmt) == {}


def test_every_local_format_resolves_against_aiter():
    mha_v4 = _aiter_mha_v4()

    for fmt in aiter_quant._Fmt:
        if fmt is aiter_quant._Fmt.NATIVE_FP8:
            # Exposed as a function, not a member: the concrete type is
            # architecture-dependent (FP8_E4M3 vs FP8_E4M3_FNUZ).
            continue
        # MXFP6 and MXFP4 are aliases and do not appear when iterating the enum,
        # so this fails if aiter ever drops them.
        assert hasattr(mha_v4.AttentionFormat, fmt.name), fmt.name


def test_every_row_is_a_format_pair_aiter_accepts():
    mha_v4 = _aiter_mha_v4()

    for fmt in aiter_quant._FORMATS:
        qk = aiter_quant._aiter_format(fmt.qk)
        v = aiter_quant._aiter_format(fmt.v)
        # Raises for a pair with no kernel row; this is the check that caught
        # MXFP4 Q/K paired with FP8 V.
        mha_v4.scale_modes_for_formats(qk, qk, v)


@pytest.fixture
def fake_aiter(monkeypatch):
    """Stand in for aiter so every format can be built and called off ROCm.

    Formats are opaque IDs to the backend -- it only has to hand aiter the same
    triple it resolved -- so the stand-in needs no real kernel.
    """

    class AttentionFormat(enum.Enum):
        BF16 = 2
        FP8_E4M3 = 3
        FP8_E4M3_FNUZ = 4
        FP6_E2M3 = 7
        FP4_E2M1 = 9
        INT8 = 10

    # aiter exposes these two as aliases, which is how _Fmt resolves them.
    AttentionFormat.MXFP6 = AttentionFormat.FP6_E2M3
    AttentionFormat.MXFP4 = AttentionFormat.FP4_E2M1

    AttentionScaleMode = enum.Enum(
        "AttentionScaleMode", sorted(set(_MXFP8_SCALE_MODE_NAMES))
    )
    calls = []

    def mha_v4(q, k, v, q_format, k_format, v_format, **kwargs):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "formats": (q_format, k_format, v_format),
                "kwargs": kwargs,
            }
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(aiter_quant, "_AITER_MHA_V4_AVAILABLE", True)
    monkeypatch.setattr(aiter_quant, "_AiterAttentionFormat", AttentionFormat)
    monkeypatch.setattr(aiter_quant, "_AiterAttentionScaleMode", AttentionScaleMode)
    monkeypatch.setattr(aiter_quant, "_aiter_mha_v4", mha_v4)
    monkeypatch.setattr(
        aiter_quant, "_aiter_native_fp8_format", lambda: AttentionFormat.FP8_E4M3
    )
    # gfx950 carries every row, so no format is skipped for arch reasons here.
    monkeypatch.setattr(aiter_quant, "is_gfx95_supported", lambda: True)
    monkeypatch.setattr(aiter_quant, "is_gfx942_supported", lambda: False)
    return types.SimpleNamespace(
        calls=calls, formats=AttentionFormat, scale_modes=AttentionScaleMode
    )


def _select_format(format_name: str | None) -> None:
    """Select the format the way a run does, through --attention-backend-config.

    The conftest fixture hands each test its own server args, so this is undone
    when the test ends.
    """
    get_global_server_args().attention_backend_config = (
        {} if format_name is None else {"format": format_name}
    )


def _build_impl(format_name: str | None, **overrides):
    _select_format(format_name)
    kwargs = {
        "num_heads": HEADS,
        "head_size": HEAD_DIM,
        "softmax_scale": SOFTMAX_SCALE,
        **overrides,
    }
    return aiter_quant.AITERQuantImpl(**kwargs)


@pytest.mark.parametrize("format_name", _FORMAT_NAMES)
def test_each_format_reaches_mha_v4_with_one_shared_qk_format(format_name, fake_aiter):
    # mha_v4 rejects a call whose Q and K formats differ, and the backend is the
    # only thing that guarantees they match -- the table stores one Q/K entry,
    # but forward passes it twice and could pass the V entry by mistake.
    impl = _build_impl(format_name)
    q, k, v = (torch.zeros(1, 8, HEADS, HEAD_DIM, dtype=torch.bfloat16),) * 3

    impl.forward(q, k, v, None)

    (call,) = fake_aiter.calls
    q_format, k_format, v_format = call["formats"]
    assert q_format is k_format
    assert q_format is impl.qk_format and v_format is impl.v_format
    assert call["kwargs"]["softmax_scale"] == SOFTMAX_SCALE


@pytest.mark.parametrize("format_name", _FORMAT_NAMES)
def test_each_format_passes_scale_modes_only_when_block_scaled(format_name, fake_aiter):
    # A canonical row that also sent scale modes, or MXFP8 that sent none, would
    # land on the wrong recipe: aiter picks MXFP8 off the scale modes alone.
    impl = _build_impl(format_name)
    q, k, v = (torch.zeros(1, 8, HEADS, HEAD_DIM, dtype=torch.bfloat16),) * 3

    impl.forward(q, k, v, None)

    (call,) = fake_aiter.calls
    sent = [key for key in call["kwargs"] if key.endswith("_scale_mode")]
    if format_name == "mxfp8":
        assert tuple(call["kwargs"][key].name for key in sent) == (
            _MXFP8_SCALE_MODE_NAMES
        )
    else:
        assert sent == []


def test_omitted_format_builds_the_default(fake_aiter):
    # The default is applied when --attention-backend-config carries no format.
    impl = _build_impl(None)
    assert impl.format == aiter_quant._DEFAULT_FORMAT


def test_rejects_the_unsupported_call_shapes(fake_aiter):
    # Every row is a head-dim-128 non-causal ASM object, so these two guards are
    # format-independent and sit ahead of format resolution in __init__.
    with pytest.raises(NotImplementedError, match="head_dim == 128"):
        _build_impl(None, head_size=64)
    with pytest.raises(NotImplementedError, match="causal"):
        _build_impl(None, causal=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
