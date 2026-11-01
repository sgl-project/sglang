# SPDX-License-Identifier: Apache-2.0
"""The aiter_quant backend: its format table, config resolution, and dispatch.

Needs the aiter import, not a GPU. Every check here is either table data or a
recorded call, and the one thing asked of aiter -- whether it accepts a row's
(Q/K, V) pair -- is plain enum logic in `scale_modes_for_formats`.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.layers.attention.backends import aiter_quant
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.test.test_utils import CustomTestCase

HEADS = 8
HEAD_DIM = 128
SEQUENCE = 8
SOFTMAX_SCALE = HEAD_DIM**-0.5

# Native FP8 is pinned to gfx950's E4M3 below, so every expectation names it.
FP8 = "FP8_E4M3"

# aiter's canonical MXFP8 recipe: FP8 operands, E8M0 1x32 Q/K scales, per-tensor
# V scale. aiter picks MXFP8 off these modes alone, not off the format pair.
MXFP8_SCALE_MODES = ("E8M0_PER_1X32", "E8M0_PER_1X32", "F32_PER_TENSOR")


@unittest.skipUnless(
    aiter_quant._AITER_MHA_V4_AVAILABLE,
    "requires SGLANG_USE_AITER=1 and an aiter that ships aiter.ops.mha_v4",
)
class _AiterQuantTestCase(CustomTestCase):
    """Real aiter, with only the parts that read the hardware stubbed.

    `native_fp8_format()` probes the arch and the ASM launch needs a GPU.
    Everything else -- the formats, the scale modes, the format contract -- is
    plain enum logic, so it stays real.
    """

    def setUp(self):
        super().setUp()
        self.calls = []
        self.addCleanup(set_global_server_args, server_args_module._global_server_args)
        set_global_server_args(SimpleNamespace(attention_backend_config=None))

        self._patch("_aiter_mha_v4", self._record_call)
        # gfx942 resolves native FP8 to E4M3_FNUZ; pinning gfx950's E4M3 keeps
        # the arch probe out of these cases and leaves the gate to its own ones.
        self._patch(
            "_aiter_native_fp8_format",
            lambda: aiter_quant._AiterAttentionFormat.FP8_E4M3,
        )
        self._patch("is_gfx95_supported", lambda: True)
        self._patch("is_gfx942_supported", lambda: False)

    def _patch(self, name, value):
        patcher = patch.object(aiter_quant, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _record_call(self, q, k, v, q_format, k_format, v_format, **kwargs):
        # What `forward` passes is the contract under test, and none of it is
        # visible in the kernel's output.
        self.calls.append(
            {
                "operands": (q, k, v),
                "formats": (q_format, k_format, v_format),
                "kwargs": kwargs,
            }
        )
        return torch.zeros_like(q)

    def _build(self, format_name=None, **overrides):
        """Build the impl the way a run does, selecting through server args."""
        aiter_quant.get_global_server_args().attention_backend_config = (
            {} if format_name is None else {"format": format_name}
        )
        kwargs = {
            "num_heads": HEADS,
            "head_size": HEAD_DIM,
            "softmax_scale": SOFTMAX_SCALE,
            **overrides,
        }
        return aiter_quant.AITERQuantImpl(**kwargs)

    def _operands(self):
        return (torch.zeros(1, SEQUENCE, HEADS, HEAD_DIM, dtype=torch.bfloat16),) * 3

    def _forward(self, format_name=None):
        self._build(format_name).forward(*self._operands(), None)
        (call,) = self.calls
        return call

    def _assert_dispatch(self, format_name, formats, scale_modes=()):
        """Run one format and check the (Q, K, V) formats and scale modes sent."""
        call = self._forward(format_name)
        self.assertEqual(tuple(fmt.name for fmt in call["formats"]), formats)
        sent = [key for key in call["kwargs"] if key.endswith("_scale_mode")]
        self.assertEqual(tuple(call["kwargs"][key].name for key in sent), scale_modes)
        self.assertEqual(call["kwargs"]["softmax_scale"], SOFTMAX_SCALE)


class TestAiterQuantFormatTable(_AiterQuantTestCase):
    def test_every_row_is_a_pair_aiter_accepts(self):
        # aiter validates the (Q/K, V) triple itself, so ask it rather than
        # restate the table: a row it has no recipe for raises here instead of
        # at the first generation that selects the format.
        from aiter.ops import mha_v4

        for fmt in aiter_quant._FORMATS:
            with self.subTest(format=fmt.name):
                qk = aiter_quant._aiter_format(fmt.qk)
                v = aiter_quant._aiter_format(fmt.v)
                mha_v4.scale_modes_for_formats(qk, qk, v)


class TestAiterQuantFormatSelection(_AiterQuantTestCase):
    def test_omitted_format_builds_the_default(self):
        self.assertIn(aiter_quant._DEFAULT_FORMAT, aiter_quant._FORMATS_BY_NAME)
        self.assertEqual(self._build().format, aiter_quant._DEFAULT_FORMAT)

    def test_format_name_is_case_insensitive(self):
        # _resolve_format lowercases, so a config written MxFp8 is not unknown.
        self.assertEqual(self._build("MxFp8").format, "mxfp8")

    def test_unknown_format_names_the_supported_ones(self):
        with self.assertRaises(ValueError) as raised:
            self._build("f4f4")
        # The message is the only place a user learns the valid names.
        for fmt in aiter_quant._FORMATS:
            self.assertIn(fmt.name, str(raised.exception))


class TestAiterQuantArchGate(_AiterQuantTestCase):
    """gfx942's fmha_v4 manifest ships a subset of gfx950's rows."""

    def _select_gfx942(self):
        self._patch("is_gfx95_supported", lambda: False)
        self._patch("is_gfx942_supported", lambda: True)

    def test_gfx942_rejects_mxfp4(self):
        # Selecting a row gfx942 lacks must raise rather than reach the
        # dispatcher and miss.
        self._select_gfx942()
        with self.assertRaisesRegex(NotImplementedError, "no gfx942 kernel row"):
            self._build("mxfp4")

    def test_gfx942_admits_the_rows_its_manifest_ships(self):
        self._select_gfx942()
        for fmt in aiter_quant._FORMATS:
            if fmt.gfx942:
                with self.subTest(format=fmt.name):
                    self.assertEqual(self._build(fmt.name).format, fmt.name)

    def test_an_arch_with_no_mha_v4_rows_is_rejected(self):
        self._patch("is_gfx95_supported", lambda: False)
        self._patch("is_gfx942_supported", lambda: False)
        with self.assertRaisesRegex(RuntimeError, "gfx950- or gfx942-class"):
            self._build()


class TestAiterQuantForward(_AiterQuantTestCase):
    """What forward hands mha_v4, one format at a time.

    mha_v4 rejects a call whose Q and K formats differ, and the backend is the
    only thing that guarantees they match -- the table stores one Q/K entry, but
    forward passes it twice and could pass the V entry by mistake. The scale
    modes are the other half: `fp8` and `mxfp8` send the same format triple, so
    the modes are the only thing that separates the two recipes.
    """

    def test_bf16fp8_sends_bf16_qk_and_native_fp8_v(self):
        self._assert_dispatch("bf16fp8", ("BF16", "BF16", FP8))

    def test_i8fp8_sends_int8_qk_and_native_fp8_v(self):
        self._assert_dispatch("i8fp8", ("INT8", "INT8", FP8))

    def test_fp8_sends_native_fp8_throughout_and_no_scale_modes(self):
        self._assert_dispatch("fp8", (FP8, FP8, FP8))

    def test_mxfp8_sends_native_fp8_throughout_with_block_scale_modes(self):
        self._assert_dispatch("mxfp8", (FP8, FP8, FP8), MXFP8_SCALE_MODES)

    def test_mxfp6_sends_fp6_e2m3_qk_and_native_fp8_v(self):
        # MXFP6 is an alias of aiter's FP6_E2M3, so the member reports that name.
        self._assert_dispatch("mxfp6", ("FP6_E2M3", "FP6_E2M3", FP8))

    def test_mxfp4_sends_fp4_e2m1_throughout(self):
        self._assert_dispatch("mxfp4", ("FP4_E2M1", "FP4_E2M1", "FP4_E2M1"))

    def test_operands_reach_the_kernel_contiguous(self):
        # The quantizers read the operands as packed BSHD; a strided view
        # reaches the kernel as the wrong elements rather than as an error.
        strided = torch.zeros(1, SEQUENCE, HEADS, HEAD_DIM * 2, dtype=torch.bfloat16)[
            ..., ::2
        ]
        self.assertFalse(strided.is_contiguous())

        self._build().forward(strided, strided, strided, None)

        (call,) = self.calls
        for operand in call["operands"]:
            self.assertTrue(operand.is_contiguous())


class TestAiterQuantRejectedCalls(_AiterQuantTestCase):
    def test_rejects_a_head_dim_other_than_128(self):
        # Every mha_v4 row is a head-dim-128 ASM object, so this guard is
        # format-independent and sits ahead of format resolution in __init__.
        with self.assertRaisesRegex(NotImplementedError, "head_dim == 128"):
            self._build(head_size=64)

    def test_rejects_causal_masking(self):
        # mha_v4 has no causal ASM row.
        with self.assertRaisesRegex(NotImplementedError, "causal"):
            self._build(causal=True)

    def test_rejects_varlen_attention(self):
        impl = self._build()
        q, k, v = self._operands()
        with self.assertRaisesRegex(NotImplementedError, "varlen"):
            impl.forward_varlen(q, k, v, cu_seqlens=None, max_seqlen=SEQUENCE)


if __name__ == "__main__":
    unittest.main()
