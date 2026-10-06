# SPDX-License-Identifier: Apache-2.0
"""The aiter_quant backend: its format table, config resolution, and dispatch.

Needs the aiter import, not a GPU. Every check here is either table data or a
recorded call, and the one thing asked of aiter -- whether it accepts a row's
(Q/K, V) pair -- is plain enum logic in `scale_modes_for_formats`.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import aiter_quant
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

pytestmark = pytest.mark.skipif(
    not aiter_quant._AITER_MHA_V4_AVAILABLE,
    reason="requires aiter.ops.mha_v4, which ships with ROCm only",
)

HEADS = 8
HEAD_DIM = 128
SEQUENCE = 8
SOFTMAX_SCALE = HEAD_DIM**-0.5

_FORMAT_NAMES = [fmt.name for fmt in aiter_quant._FORMATS]

# aiter's canonical MXFP8 recipe: FP8 operands, E8M0 1x32 Q/K scales, per-tensor
# V scale. aiter picks MXFP8 off these modes alone, not off the format pair.
_MXFP8_SCALE_MODE_NAMES = (
    "E8M0_PER_1X32",
    "E8M0_PER_1X32",
    "F32_PER_TENSOR",
)


@pytest.fixture
def aiter_without_device(monkeypatch):
    """Real aiter, with only the parts that read the hardware stubbed.

    `native_fp8_format()` probes the arch and the ASM launch needs a GPU.
    Everything else -- the formats, the scale modes, the format contract -- is
    plain enum logic, so it stays real.

    Yields the recorded calls. What `forward` passes is the contract under
    test, and none of it is visible in the kernel's output.
    """
    calls = []

    def record_call(q, k, v, q_format, k_format, v_format, **kwargs):
        calls.append(
            {
                "operands": (q, k, v),
                "formats": (q_format, k_format, v_format),
                "kwargs": kwargs,
            }
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(aiter_quant, "_aiter_mha_v4", record_call)
    # gfx942 resolves native FP8 to E4M3_FNUZ; pinning gfx950's E4M3 keeps the
    # arch probe out of these tests and leaves the gate to its own two.
    monkeypatch.setattr(
        aiter_quant,
        "_aiter_native_fp8_format",
        lambda: aiter_quant._AiterAttentionFormat.FP8_E4M3,
    )
    monkeypatch.setattr(aiter_quant, "is_gfx95_supported", lambda: True)
    monkeypatch.setattr(aiter_quant, "is_gfx942_supported", lambda: False)
    return calls


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


def _operands():
    return (torch.zeros(1, SEQUENCE, HEADS, HEAD_DIM, dtype=torch.bfloat16),) * 3


# ===== The table against aiter =====


def test_every_operand_encoding_resolves_against_aiter():
    # _aiter_format resolves _Fmt members by name, so each one must still name a
    # member or alias of aiter's enum. MXFP6 and MXFP4 are aliases and do not
    # show up when iterating it, so this asks by name rather than by iteration.
    from aiter.ops import mha_v4

    for fmt in aiter_quant._Fmt:
        if fmt is aiter_quant._Fmt.NATIVE_FP8:
            # Exposed as a function, not a member: the concrete type is
            # architecture-dependent (FP8_E4M3 vs FP8_E4M3_FNUZ).
            continue
        assert hasattr(mha_v4.AttentionFormat, fmt.name), fmt.name


def test_every_row_is_a_pair_aiter_accepts(aiter_without_device):
    # aiter validates the (Q/K, V) triple itself, so ask it rather than restate
    # the table: a row it has no recipe for raises here instead of at the first
    # generation that selects the format.
    from aiter.ops import mha_v4

    for fmt in aiter_quant._FORMATS:
        qk = aiter_quant._aiter_format(fmt.qk)
        v = aiter_quant._aiter_format(fmt.v)
        mha_v4.scale_modes_for_formats(qk, qk, v)


def test_mxfp8_is_the_only_block_scaled_row_and_uses_aiter_recipe():
    block_scaled = [fmt.name for fmt in aiter_quant._FORMATS if fmt.block_scaled]
    assert block_scaled == ["mxfp8"]

    # Resolves against aiter's real AttentionScaleMode, so a renamed member
    # fails here rather than at the first block-scaled call.
    kwargs = aiter_quant._scale_mode_kwargs(aiter_quant._FORMATS_BY_NAME["mxfp8"])
    assert list(kwargs) == ["q_scale_mode", "k_scale_mode", "v_scale_mode"]
    assert tuple(mode.name for mode in kwargs.values()) == _MXFP8_SCALE_MODE_NAMES


def test_non_block_scaled_rows_pass_no_scale_modes():
    # aiter derives the canonical modes from the format pair and rejects a
    # non-canonical triple, so these rows must leave them unset.
    for fmt in aiter_quant._FORMATS:
        if not fmt.block_scaled:
            assert aiter_quant._scale_mode_kwargs(fmt) == {}


# ===== Selecting a format through --attention-backend-config =====


def test_omitted_format_builds_the_default(aiter_without_device):
    assert aiter_quant._DEFAULT_FORMAT in aiter_quant._FORMATS_BY_NAME
    assert _build_impl(None).format == aiter_quant._DEFAULT_FORMAT


@pytest.mark.parametrize("format_name", _FORMAT_NAMES)
def test_each_format_name_selects_its_own_row(format_name, aiter_without_device):
    impl = _build_impl(format_name)
    row = aiter_quant._FORMATS_BY_NAME[format_name]

    assert impl.format == format_name
    assert impl.qk_format is aiter_quant._aiter_format(row.qk)
    assert impl.v_format is aiter_quant._aiter_format(row.v)


def test_format_name_is_case_insensitive(aiter_without_device):
    # _resolve_format lowercases, so a config written FP8 must not be unknown.
    assert _build_impl("MxFp8").format == "mxfp8"


def test_unknown_format_names_the_supported_ones():
    with pytest.raises(ValueError, match="Unknown aiter_quant format 'f4f4'") as err:
        aiter_quant._lookup_format("f4f4")
    # The message is the only place a user learns the valid names.
    for name in _FORMAT_NAMES:
        assert name in str(err.value)


# ===== The arch gate =====


def test_gfx942_admits_only_its_own_rows():
    # gfx942's fmha_v4 manifest is a subset, so selecting a row it lacks must
    # raise rather than reach the dispatcher and miss.
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


def test_gfx950_admits_every_row():
    with patch.object(aiter_quant, "is_gfx95_supported", return_value=True):
        for fmt in aiter_quant._FORMATS:
            aiter_quant._validate_arch(fmt)


def test_neither_arch_raises():
    with (
        patch.object(aiter_quant, "is_gfx95_supported", return_value=False),
        patch.object(aiter_quant, "is_gfx942_supported", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="gfx950- or gfx942-class"):
            aiter_quant._validate_arch(aiter_quant._FORMATS[0])


# ===== What forward hands aiter =====


@pytest.mark.parametrize("format_name", _FORMAT_NAMES)
def test_forward_sends_one_shared_qk_format(format_name, aiter_without_device):
    # mha_v4 rejects a call whose Q and K formats differ, and the backend is the
    # only thing that guarantees they match -- the table stores one Q/K entry,
    # but forward passes it twice and could pass the V entry by mistake.
    impl = _build_impl(format_name)

    impl.forward(*_operands(), None)

    (call,) = aiter_without_device
    q_format, k_format, v_format = call["formats"]
    assert q_format is k_format is impl.qk_format
    assert v_format is impl.v_format
    assert call["kwargs"]["softmax_scale"] == SOFTMAX_SCALE


@pytest.mark.parametrize("format_name", _FORMAT_NAMES)
def test_forward_sends_scale_modes_only_for_the_block_scaled_row(
    format_name, aiter_without_device
):
    # A canonical row that also sent scale modes, or MXFP8 that sent none, would
    # land on the wrong recipe: aiter picks MXFP8 off the scale modes alone.
    impl = _build_impl(format_name)

    impl.forward(*_operands(), None)

    (call,) = aiter_without_device
    sent = [key for key in call["kwargs"] if key.endswith("_scale_mode")]
    if aiter_quant._FORMATS_BY_NAME[format_name].block_scaled:
        assert tuple(call["kwargs"][key].name for key in sent) == (
            _MXFP8_SCALE_MODE_NAMES
        )
    else:
        assert sent == []


def test_forward_makes_operands_contiguous(aiter_without_device):
    # The quantizers read the operands as packed BSHD; a strided view reaches
    # the kernel as the wrong elements rather than as an error.
    impl = _build_impl(None)
    strided = torch.zeros(1, SEQUENCE, HEADS, HEAD_DIM * 2, dtype=torch.bfloat16)[
        ..., ::2
    ]
    assert not strided.is_contiguous()

    impl.forward(strided, strided, strided, None)

    (call,) = aiter_without_device
    for operand in call["operands"]:
        assert operand.is_contiguous()


# ===== Calls the backend refuses =====


def test_rejects_the_unsupported_call_shapes():
    # Every row is a head-dim-128 non-causal ASM object, so these two guards are
    # format-independent and sit ahead of format resolution in __init__.
    with pytest.raises(NotImplementedError, match="head_dim == 128"):
        _build_impl(None, head_size=64)
    with pytest.raises(NotImplementedError, match="causal"):
        _build_impl(None, causal=True)


def test_varlen_is_not_supported(aiter_without_device):
    impl = _build_impl(None)
    q, k, v = _operands()

    with pytest.raises(NotImplementedError, match="varlen"):
        impl.forward_varlen(q, k, v, cu_seqlens=None, max_seqlen=SEQUENCE)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
