"""Capability contract for the varlen absorbed-MLA extend path.

A subclass that swaps the decode kernel must state whether it can also serve a
ragged query, rather than inherit the base class' answer: the ragged path is
_run_varlen_absorbed_kernel(), and supports_varlen_absorbed_mla decides whether
forward_extend() reaches it at all.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    varlen_absorbed_mla_shape_ok,
    varlen_absorbed_mla_supported,
)
from sglang.srt.utils import FP4_KV_CACHE_DTYPES
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-large")


def _skip_unless_fp4_dtype_available(test_case):
    try:
        torch.float4_e2m1fn_x2
    except AttributeError:
        test_case.skipTest("torch build has no float4_e2m1fn_x2")


class TestVarlenAbsorbedMLAGate(CustomTestCase):
    # In-tree backends that swap the decode kernel. Pinned so the scan below
    # cannot pass vacuously when an import above is dropped, and so that adding
    # a backend forces a conscious answer on supports_varlen_absorbed_mla.
    DECODE_KERNEL_OVERRIDERS = (TokenspeedMLABackend, CuteDslMLABackend)

    def test_base_backend_supports_varlen(self):
        self.assertTrue(TRTLLMMLABackend.supports_varlen_absorbed_mla)

    def test_tokenspeed_opts_out(self):
        # It inherits backend == "trtllm-gen", so the opt-out must be explicit.
        self.assertFalse(TokenspeedMLABackend.supports_varlen_absorbed_mla)
        self.assertTrue(issubclass(TokenspeedMLABackend, TRTLLMMLABackend))

    def test_cutedsl_opts_out(self):
        self.assertFalse(CuteDslMLABackend.supports_varlen_absorbed_mla)
        self.assertTrue(issubclass(CuteDslMLABackend, TRTLLMMLABackend))

    def test_opted_out_backends_have_no_ragged_kernel(self):
        # Premise of both opt-outs: neither brings its own ragged path, so the
        # capability they decline really is unimplemented rather than declared
        # away. If either grows one, revisit its flag.
        for cls in self.DECODE_KERNEL_OVERRIDERS:
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("_run_varlen_absorbed_kernel", cls.__dict__)

    def test_shipped_subclasses_declare_support_explicitly(self):
        # The defect was a subclass nobody had inventoried. Any shipped subclass
        # that swaps the decode kernel must state its choice rather than inherit
        # one. Test-local subclasses are excluded so this stays order-independent.
        found = set()
        for cls in TRTLLMMLABackend.__subclasses__():
            if not cls.__module__.startswith("sglang."):
                continue
            if "_run_decode_kernel" not in cls.__dict__:
                continue
            with self.subTest(cls=cls.__name__):
                self.assertIn("supports_varlen_absorbed_mla", cls.__dict__)
            found.add(cls)
        self.assertEqual(
            found,
            set(self.DECODE_KERNEL_OVERRIDERS),
            "the set of in-tree decode-kernel overrides changed; update "
            "DECODE_KERNEL_OVERRIDERS after deciding the new class' "
            "supports_varlen_absorbed_mla",
        )


class TestVarlenAbsorbedCapabilityContract(CustomTestCase):
    """ServerArgs decides whether the prefill CUDA graph may stay on
    tc_piecewise for trtllm_mla; forward_extend decides whether the ragged
    absorbed path actually runs. If those two ever answer differently, the graph
    is captured and the extend silently falls back to paged MLA -- the exact
    regression this path removes, with nothing raised to notice it by. They must
    therefore read one predicate, not two copies of it.
    """

    _BACKEND = "sglang.srt.layers.attention.trtllm_mla_backend"

    def _server_args(self):
        from sglang.srt.server_args import ServerArgs

        return ServerArgs(model_path="dummy")

    def test_server_args_delegates_to_the_backend_predicate(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            trtllm_mla_has_varlen_absorbed,
        )

        args = self._server_args()
        for supported in (True, False):
            with self.subTest(supported=supported):
                with (
                    patch(
                        "sglang.srt.arg_groups.overrides.attention_backends_of",
                        return_value=("trtllm_mla", "trtllm_mla"),
                    ),
                    patch(
                        f"{self._BACKEND}.varlen_absorbed_mla_supported",
                        return_value=supported,
                    ) as helper,
                ):
                    has = trtllm_mla_has_varlen_absorbed(args)
                self.assertEqual(has, supported)
                helper.assert_called_once_with(args.kv_cache_dtype)

    def test_other_backends_are_never_excluded(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            trtllm_mla_has_varlen_absorbed,
        )

        args = self._server_args()
        with patch(
            "sglang.srt.arg_groups.overrides.attention_backends_of",
            return_value=("fa3", "fa3"),
        ):
            self.assertTrue(trtllm_mla_has_varlen_absorbed(args))

    def test_fp4_kv_spellings_match_the_dtype_resolver(self):
        # A --kv-cache-dtype spelling that resolves to the packed 4-bit dtype but
        # is missing from FP4_KV_CACHE_DTYPES would let ServerArgs upgrade to
        # tc_piecewise for a config forward_extend refuses to serve.
        from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype

        _skip_unless_fp4_dtype_available(self)

        def resolve(name):
            _, dtype = configure_kv_cache_dtype(
                server_args_kv_cache_dtype=name,
                model=SimpleNamespace(quant_config=None),
                model_dtype=torch.bfloat16,
                is_draft_worker=False,
                is_dflash=False,
                speculative_draft_attention_backend="",
            )
            return dtype

        for name in FP4_KV_CACHE_DTYPES:
            with self.subTest(name=name):
                self.assertIs(resolve(name), torch.float4_e2m1fn_x2)
        for name in ("fp8_e4m3", "bf16"):
            with self.subTest(name=name):
                self.assertIsNot(resolve(name), torch.float4_e2m1fn_x2)

    def test_string_and_dtype_forms_agree(self):
        # ServerArgs passes the CLI string, the backend passes a torch dtype.
        _skip_unless_fp4_dtype_available(self)
        with patch(f"{self._BACKEND}.is_sm100_supported", return_value=True):
            self.assertFalse(varlen_absorbed_mla_supported("nvfp4"))
            self.assertFalse(varlen_absorbed_mla_supported(torch.float4_e2m1fn_x2))
            self.assertTrue(varlen_absorbed_mla_supported("fp8_e4m3"))
            self.assertTrue(varlen_absorbed_mla_supported(torch.float8_e4m3fn))

    def test_non_sm10_is_unsupported_whatever_the_kv_dtype(self):
        with patch(f"{self._BACKEND}.is_sm100_supported", return_value=False):
            self.assertFalse(varlen_absorbed_mla_supported("fp8_e4m3"))
            self.assertFalse(varlen_absorbed_mla_supported(torch.float8_e4m3fn))


class TestVarlenAbsorbedMLAShapeGate(CustomTestCase):
    """varlen_absorbed_mla_shape_ok() mirrors flashinfer 0.6.17's own trtllm-gen
    MLA decode dispatch conditions (mla/_core.py, trtllm_batch_decode_with_kv_cache_mla).
    _run_varlen_absorbed_kernel has no cute-dsl fallback, so a shape flashinfer
    would silently redirect away from trtllm-gen must be excluded before that
    call, not discovered by it. Kimi-K3 under wide-EP DP-attention
    (num_attention_heads=96, attn_tp_size=1) is the regression this closes.
    """

    def test_num_heads_q_exclusion_boundaries(self):
        # Strict inequality: 64 and 128 themselves are outside the exclusion.
        cases = [
            (12, True),  # Case18 plain TP8 on Kimi-K3 (96 // 8)
            (64, True),  # boundary, not excluded
            (65, False),
            (96, False),  # Kimi-K3 DEP16 (96 // 1); the crash this test pins
            (127, False),
            (128, True),  # boundary, not excluded
            (256, True),
        ]
        for num_heads_q, expected in cases:
            with self.subTest(num_heads_q=num_heads_q):
                self.assertEqual(
                    varlen_absorbed_mla_shape_ok(num_heads_q, page_size=64),
                    expected,
                )

    def test_page_size_must_be_32_or_64(self):
        for page_size, expected in ((32, True), (64, True), (16, False), (128, False)):
            with self.subTest(page_size=page_size):
                self.assertEqual(
                    varlen_absorbed_mla_shape_ok(num_heads_q=12, page_size=page_size),
                    expected,
                )

    def test_either_condition_alone_excludes(self):
        # A shape can fail on num_heads_q, page_size, or both -- any one is
        # enough to exclude it, not both required.
        self.assertFalse(varlen_absorbed_mla_shape_ok(96, page_size=64))
        self.assertFalse(varlen_absorbed_mla_shape_ok(12, page_size=16))
        self.assertFalse(varlen_absorbed_mla_shape_ok(96, page_size=16))


if __name__ == "__main__":
    unittest.main()
