"""Capability and dispatch contract for varlen absorbed-MLA extend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _get_cute_dsl_workspace_buffer,
    _get_varlen_absorbed_workspace_buffer,
    configured_varlen_absorbed_mla_supported,
    varlen_absorbed_mla_supported,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import FP4_KV_CACHE_DTYPES
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _skip_unless_fp4_dtype_available(test_case):
    try:
        torch.float4_e2m1fn_x2
    except AttributeError:
        test_case.skipTest("torch build has no float4_e2m1fn_x2")


class TestVarlenAbsorbedMLAGate(CustomTestCase):
    DECODE_KERNEL_OVERRIDERS = (TokenspeedMLABackend, CuteDslMLABackend)

    def test_base_backend_owns_varlen_extend(self):
        self.assertTrue(TRTLLMMLABackend.owns_varlen_absorbed_extend)

    def test_tokenspeed_does_not_own_varlen_extend(self):
        self.assertFalse(TokenspeedMLABackend.owns_varlen_absorbed_extend)
        self.assertTrue(issubclass(TokenspeedMLABackend, TRTLLMMLABackend))

    def test_decode_only_cutedsl_does_not_own_varlen_extend(self):
        self.assertFalse(CuteDslMLABackend.owns_varlen_absorbed_extend)
        self.assertTrue(issubclass(CuteDslMLABackend, TRTLLMMLABackend))

    def test_decode_overrides_do_not_own_a_varlen_extend_kernel(self):
        for cls in self.DECODE_KERNEL_OVERRIDERS:
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("_run_varlen_absorbed_kernel", cls.__dict__)

    def test_shipped_decode_overrides_declare_ownership_explicitly(self):
        found = set()
        for cls in TRTLLMMLABackend.__subclasses__():
            if not cls.__module__.startswith("sglang."):
                continue
            if "_run_decode_kernel" not in cls.__dict__:
                continue
            with self.subTest(cls=cls.__name__):
                self.assertIn("owns_varlen_absorbed_extend", cls.__dict__)
            found.add(cls)
        self.assertEqual(
            found,
            set(self.DECODE_KERNEL_OVERRIDERS),
            "the set of in-tree decode-kernel overrides changed; update "
            "DECODE_KERNEL_OVERRIDERS after deciding the new class' "
            "owns_varlen_absorbed_extend",
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

    def _server_args(self, *, kv_cache_dtype="fp8_e4m3", model_dtype=torch.bfloat16):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy", kv_cache_dtype=kv_cache_dtype)
        args._model_config = SimpleNamespace(dtype=model_dtype)
        return args

    def test_server_args_delegates_to_the_backend_predicate(self):
        from sglang.srt.arg_groups.cuda_graph_hook import trtllm_mla_has_varlen_absorbed

        args = self._server_args()
        for supported in (True, False):
            with self.subTest(supported=supported):
                with (
                    patch(
                        "sglang.srt.arg_groups.overrides.attention_backends_of",
                        return_value=("trtllm_mla", "trtllm_mla"),
                    ),
                    patch(
                        f"{self._BACKEND}.configured_varlen_absorbed_mla_supported",
                        return_value=supported,
                    ) as helper,
                ):
                    has = trtllm_mla_has_varlen_absorbed(args)
                self.assertEqual(has, supported)
                helper.assert_called_once_with(args.kv_cache_dtype, torch.bfloat16)

    def test_auto_kv_uses_resolved_model_dtype(self):
        from sglang.srt.arg_groups.cuda_graph_hook import trtllm_mla_has_varlen_absorbed

        for model_dtype, expected in (
            (torch.bfloat16, True),
            (torch.float16, False),
        ):
            with self.subTest(model_dtype=model_dtype):
                args = self._server_args(kv_cache_dtype="auto", model_dtype=model_dtype)
                with (
                    patch(
                        "sglang.srt.arg_groups.overrides.attention_backends_of",
                        return_value=("trtllm_mla", "trtllm_mla"),
                    ),
                    patch(f"{self._BACKEND}.is_sm100_supported", return_value=True),
                ):
                    self.assertEqual(trtllm_mla_has_varlen_absorbed(args), expected)

    def test_configured_auto_and_explicit_forms_match_runtime_gate(self):
        with patch(f"{self._BACKEND}.is_sm100_supported", return_value=True):
            self.assertTrue(
                configured_varlen_absorbed_mla_supported("auto", torch.bfloat16)
            )
            self.assertFalse(
                configured_varlen_absorbed_mla_supported("auto", torch.float16)
            )
            self.assertTrue(
                configured_varlen_absorbed_mla_supported("fp8_e4m3", torch.float16)
            )

    def test_other_backends_are_never_excluded(self):
        from sglang.srt.arg_groups.cuda_graph_hook import trtllm_mla_has_varlen_absorbed

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

    def test_only_supported_string_and_dtype_forms_are_allowed(self):
        with patch(f"{self._BACKEND}.is_sm100_supported", return_value=True):
            for dtype in (
                "bf16",
                "bfloat16",
                "fp8_e4m3",
                torch.bfloat16,
                torch.float8_e4m3fn,
            ):
                with self.subTest(dtype=dtype):
                    self.assertTrue(varlen_absorbed_mla_supported(dtype))
            for dtype in (
                "auto",
                "fp8_e5m2",
                "mxfp8",
                "nvfp4",
                "fp4_mx_block16",
                torch.float16,
                torch.float8_e5m2,
            ):
                with self.subTest(dtype=dtype):
                    self.assertFalse(varlen_absorbed_mla_supported(dtype))

            try:
                fp4_dtype = torch.float4_e2m1fn_x2
            except AttributeError:
                pass
            else:
                self.assertFalse(varlen_absorbed_mla_supported(fp4_dtype))

    def test_non_sm10_is_unsupported_whatever_the_kv_dtype(self):
        with patch(f"{self._BACKEND}.is_sm100_supported", return_value=False):
            self.assertFalse(varlen_absorbed_mla_supported("fp8_e4m3"))
            self.assertFalse(varlen_absorbed_mla_supported(torch.float8_e4m3fn))


class TestVarlenAbsorbedMLARouting(CustomTestCase):
    _BACKEND = "sglang.srt.layers.attention.trtllm_mla_backend"

    def _assert_paged_fallback(self, *, dcp_enabled, skip_softmax):
        backend = object.__new__(TRTLLMMLABackend)
        backend.backend = "trtllm-gen"
        backend.disable_chunked_prefix_cache = False
        backend._varlen_absorbed_arch_dtype_ok = True
        backend._kv_shard_pool = None

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens_cpu=[0],
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[3],
            seq_lens=torch.tensor([3], dtype=torch.int32),
            spec_info=None,
        )

        with (
            patch(f"{self._BACKEND}.is_in_tc_piecewise_cuda_graph", return_value=True),
            patch(f"{self._BACKEND}.is_in_breakable_cuda_graph", return_value=False),
            patch(
                f"{self._BACKEND}.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
            ),
            patch(
                f"{self._BACKEND}.envs."
                "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get",
                return_value=skip_softmax,
            ),
            patch.object(
                FlashInferMLAAttnBackend, "init_forward_metadata"
            ) as paged_fallback,
        ):
            backend.init_forward_metadata(forward_batch)

        paged_fallback.assert_called_once_with(forward_batch)
        self.assertTrue(backend.forward_prefill_metadata.fallback_to_flashinfer_impl)
        self.assertIsNone(backend.forward_prefill_metadata.block_kv_indices)

    def test_dcp_keeps_the_paged_fallback(self):
        self._assert_paged_fallback(dcp_enabled=True, skip_softmax=None)

    def test_skip_softmax_keeps_the_paged_fallback(self):
        self._assert_paged_fallback(dcp_enabled=False, skip_softmax=1.0)


class TestVarlenAbsorbedMLADispatchContract(CustomTestCase):
    def _backend(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.backend = "trtllm-gen"
        backend.workspace_buffer = sentinel.workspace_buffer
        backend._varlen_absorbed_workspace_buffer = sentinel.varlen_workspace_buffer
        backend._multi_ctas_kv_counter_buffer = sentinel.counter_buffer
        backend.qk_nope_head_dim = 128
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend._compute_decode_bmm1_scale = MagicMock(return_value=1.0)
        return backend

    def _call(self, backend, *, varlen, return_lse=False):
        kwargs = {}
        if varlen:
            kwargs.update(
                cum_seq_lens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
                max_q_len=2,
            )
        with patch(
            "sglang.srt.layers.attention.trtllm_mla_backend.flashinfer.decode."
            "trtllm_batch_decode_with_kv_cache_mla",
            return_value=sentinel.output,
        ) as kernel:
            output = backend._call_trtllm_batch_decode_mla(
                query=torch.empty(3, 96, 576),
                kv_cache=torch.empty(1),
                block_tables=torch.empty(1),
                seq_lens=torch.tensor([1, 1], dtype=torch.int32),
                max_seq_len=1,
                layer=SimpleNamespace(),
                return_lse=return_lse,
                **kwargs,
            )
        self.assertIs(output, sentinel.output)
        return kernel.call_args.kwargs

    def test_varlen_call_allows_flashinfer_auto_dispatch(self):
        kwargs = self._call(self._backend(), varlen=True)
        self.assertNotIn("backend", kwargs)
        self.assertNotIn("multi_ctas_kv_counter_buffer", kwargs)
        self.assertIs(kwargs["workspace_buffer"], sentinel.varlen_workspace_buffer)
        self.assertEqual(kwargs["max_q_len"], 2)
        self.assertTrue(torch.equal(kwargs["cum_seq_lens_q"], torch.tensor([0, 2, 3])))

    def test_dense_call_keeps_explicit_trt_counter(self):
        kwargs = self._call(self._backend(), varlen=False)
        self.assertEqual(kwargs["backend"], "trtllm-gen")
        self.assertIs(kwargs["multi_ctas_kv_counter_buffer"], sentinel.counter_buffer)
        self.assertIs(kwargs["workspace_buffer"], sentinel.workspace_buffer)

    def test_cute_and_varlen_workspaces_are_separate_and_shared(self):
        buffers = {}

        def get_buffer(name, factory):
            if name not in buffers:
                buffers[name] = factory()
            return buffers[name]

        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mla_backend.get_buffer",
                side_effect=get_buffer,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mla_backend.torch.zeros",
                side_effect=[sentinel.cute_workspace, sentinel.varlen_workspace],
            ) as zeros,
        ):
            first = _get_cute_dsl_workspace_buffer(1024, torch.device("cuda"))
            second = _get_cute_dsl_workspace_buffer(1024, torch.device("cuda"))
            varlen_first = _get_varlen_absorbed_workspace_buffer(
                1024, torch.device("cuda")
            )
            varlen_second = _get_varlen_absorbed_workspace_buffer(
                1024, torch.device("cuda")
            )

        self.assertIs(first, sentinel.cute_workspace)
        self.assertIs(second, first)
        self.assertIs(varlen_second, varlen_first)
        self.assertIsNot(varlen_first, first)
        self.assertEqual(zeros.call_count, 2)

    def test_return_lse_is_forwarded_to_flashinfer(self):
        for return_lse in (False, True):
            with self.subTest(return_lse=return_lse):
                kwargs = self._call(
                    self._backend(), varlen=False, return_lse=return_lse
                )
                self.assertIs(kwargs["return_lse"], return_lse)

    def test_run_decode_kernel_forwards_return_lse(self):
        backend = self._backend()
        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mla_backend.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=True),
            ),
            patch.object(
                backend,
                "_call_trtllm_batch_decode_mla",
                return_value=(sentinel.output, sentinel.lse),
            ) as kernel,
        ):
            output = backend._run_decode_kernel(
                query=torch.empty(2, 96, 576),
                kv_cache=torch.empty(1),
                block_tables=torch.empty(1),
                seq_lens=torch.tensor([1, 1], dtype=torch.int32),
                max_seq_len=1,
                layer=SimpleNamespace(),
                return_lse=True,
            )

        self.assertEqual(output, (sentinel.output, sentinel.lse))
        self.assertIs(kernel.call_args.kwargs["return_lse"], True)


if __name__ == "__main__":
    unittest.main()
